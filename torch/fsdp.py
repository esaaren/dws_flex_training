# Copyright 2025 Google Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



###########################
####### IMPORTS
###########################

import functools
import logging
import os
import time
import datetime 
import pandas as pd
from datasets import Dataset

import datasets 
from datasets import load_dataset

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dc
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, ShardingStrategy
from torch.distributed.fsdp.wrap import (_or_policy, 
                                         lambda_auto_wrap_policy,
                                         transformer_auto_wrap_policy)
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload #

from peft import get_peft_model, LoraConfig, TaskType
from tqdm.auto import tqdm
import transformers # Base import 
from transformers import (AutoConfig, AutoTokenizer, LlamaForCausalLM, 
                          get_linear_schedule_with_warmup)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer 


torch.cuda.memory._record_memory_history(
    max_entries=100000
)

logging.basicConfig(level=logging.INFO)

###########################
####### ENV VARS
###########################
os.environ["TOKENIZERS_PARALLELISM"] = "true"
data_path = os.environ.get('DATA_PATH', '/home/esaarenvirta/training_data')
ckpt_path = os.environ.get('CHECKPOINT_PATH', '/home/esaarenvirta/checkpoints')
data_split_train=os.environ.get('DATA_SPLIT_TRAIN', 'train')
data_split_eval=os.environ.get('DATA_SPLIT_EVAL', 'validation')
data_subset=os.environ.get('DATA_SUBSET', 'split')
dataset_name=os.environ.get('DATASET_NAME', "dair-ai/emotion")
max_length=int(os.environ.get('SEQUENCE_LENGTH', 315)) 
tokenizer_batch_size=int(os.environ.get('TOKENIZER_BATCH_SIZE', 10000))
batch_size=int(os.environ.get('BATCH_SIZE', 8))
num_epochs=int(os.environ.get('EPOCHS', 10))
accumulation_steps=int(os.environ.get('GRAD_ACC_STEPS', 1))
model_name=os.environ.get('MODEL_NAME', "meta-llama/Llama-3.1-8B")
peft=os.environ.get('USE_PEFT', 'False')
if peft == 'False':
    peft = False
else:
    peft = True
checkpoint_epochs=int(os.environ.get('CHECKPOINT_EPOCHS', 10)) 
tokenizer_num_proc=int(os.environ.get('TOKENIZER_NUM_PROC', 100))
reload_checkpoint=os.environ.get('RELOAD_CHECKPOINT', None)
nccl_timeout=int(os.environ.get('NCCL_TIMEOUT', 120))
if reload_checkpoint == "None":
    reload_checkpoint = None
batch_log_interval = int(os.environ.get('BATCH_LOG_INTERVAL', 100))
grad_checkpointing = os.environ.get('GRAD_CHECKPOINTING_ENABLE', 'False')

###########################
####### PYTORCH AUTOWRAP POLICY
###########################
def fsdp_auto_wrap_policy(model, transformer_layer_names):
    import functools

    from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy, transformer_auto_wrap_policy

    def lambda_policy_fn(module):
        if (
            len(list(module.named_children())) == 0
            and getattr(module, "weight", None) is not None
            and module.weight.requires_grad
        ):
            return True
        return False

    lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
    transformer_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=set(transformer_layer_names)
    )

    auto_wrap_policy = functools.partial(_or_policy, policies=[lambda_policy, transformer_wrap_policy])
    return auto_wrap_policy


###########################
####### MAIN TRAINING FUNCTION
###########################
def train_fn():

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    desired_timeout = datetime.timedelta(minutes=nccl_timeout) 

    dist.init_process_group("nccl", timeout=desired_timeout) 

    rank = dist.get_rank()

    logging.info(f"Running DDP on rank {rank}, local rank {local_rank} with world size {world_size}")
    torch.cuda.set_device(local_rank)
    logging.info(f"Inside rank  {rank} {local_rank}")

    ###########################
    ####### Model + Tokenizer Download 
    ###########################

    # Download model / tokenizer on rank 0 so no deadlocks
    if rank == 0:
        logging.info(f"Rank {rank} downloading model and config...")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        _ = LlamaForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16)
        logging.info(f"Rank {rank} finished download/cache check.")

    # Barrier after downloading
    dist.barrier(device_ids=[local_rank])
    logging.info(f"Rank {rank} loading model from cache...")

    # Tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenizer and text processing functions 
    # Pre-process the data so we create an instruction prompt
    def add_prompt(text):
        prompt = f"Classify the emotion of this text: {text}, Emotion: "
        return prompt

    # Input col needs to be called "text"
    def process_text(examples):
        return tokenizer(examples['text'], truncation=True, max_length=max_length, padding='max_length',
                        return_tensors='pt')


    ###########################
    ####### FSDP v1 Implementation TODO: Upgrade to v2 
    ###########################
    config = AutoConfig.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(
            model_name,
            device_map=(
                None # By not specifying a device_map, the entire model is initially loaded onto the CPU 
            ),
            torch_dtype=torch.bfloat16 #"auto",
    )

    # Optional PEFT and log trainable params 
    if peft is True:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False, r=16, lora_alpha=32, lora_dropout=0.05
        )
        model = get_peft_model(model, peft_config)
        logging.info(model.print_trainable_parameters())


    # Auto wrap policy init
    my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, [LlamaDecoderLayer]) # Wrap the layer that wraps the attention blocks
    
    if grad_checkpointing == 'True':
        model.gradient_checkpointing_enable()

    model = FSDP(
            model,
            auto_wrap_policy=(
                my_auto_wrapping_policy 
            ),
            cpu_offload=( # allows parameters that are not currently being used in a computation to be offloaded to the CPU, saving GPU memory.
                CPUOffload(offload_params=False)
            ),
            mixed_precision=
                 None, # TODO: Implement
            sharding_strategy=ShardingStrategy.FULL_SHARD, # It shards the parameters, gradients, and optimizer states across all GPUs.
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True, # Reduces the number of all-gather operations, which can be expensive.
            sync_module_states=False, 
            param_init_fn=( 
                (
                    lambda module: module.to_empty(
                        device=torch.device("cuda"), recurse=False
                    )
                )
                if False and rank != 0
                else None
            ),
            use_orig_params=True, 
        )
    
    # Barrier after model loading 
    dist.barrier(device_ids=[local_rank])


    ###########################
    ####### Data Processing
    ###########################
    
    # Load from disk if data is already there 
    if os.path.isdir(data_path):
        logging.info('Training data already processed and on storage!')
        # load after processed 
        loaded_tokenized_dataset_train = datasets.load_from_disk(data_path + "/train")
        loaded_tokenized_dataset_eval = datasets.load_from_disk(data_path + "/eval")

        loaded_tokenized_dataset_train.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        loaded_tokenized_dataset_eval.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    else:
        # Process data only on rank 0 to avoid redundant work
        if rank == 0:
            print('No training data found, downloading and processing then saving!')
            
            # TODO: Dataset is technically still hardcoded to dair-ai/emotion via ENV vars 
            dataset_train = load_dataset(dataset_name, data_subset, split=data_split_train)
            dataset_eval = load_dataset(dataset_name, data_subset, split=data_split_eval)
            df_train = pd.DataFrame(dataset_train)
            df_eval = pd.DataFrame(dataset_eval)

            # map the labels 
            label_mapping = {
                0: 'sadness',
                1: 'joy',
                2: 'love',
                3: 'anger',
                4: 'fear',
                5: 'surprise'
            }
            
            # Format the prompt and then apply the str version of the labels 
            df_train['text'] = df_train['text'].apply(add_prompt)
            df_train['label_text'] = df_train['label'].map(label_mapping)
            df_train['text'] = df_train['text'] + df_train['label_text']

            df_eval['text'] = df_eval['text'].apply(add_prompt)
            df_eval['label_text'] = df_eval['label'].map(label_mapping)
            df_eval['text'] = df_eval['text'] + df_eval['label_text']

            dataset_train = Dataset.from_pandas(df_train)
            dataset_eval = Dataset.from_pandas(df_eval)

            tokenized_dataset_train = dataset_train.map(process_text, batch_size=tokenizer_batch_size, num_proc=tokenizer_num_proc, batched=True)
            tokenized_dataset_train.set_format(type='torch', columns=['input_ids', 'attention_mask']) 
            tokenized_dataset_train.save_to_disk(data_path + '/train')
            del tokenized_dataset_train

            tokenized_dataset_eval= dataset_eval.map(process_text, batch_size=tokenizer_batch_size, num_proc=tokenizer_num_proc, batched=True)
            tokenized_dataset_eval.set_format(type='torch', columns=['input_ids', 'attention_mask']) 
            tokenized_dataset_eval.save_to_disk(data_path + '/eval')
            del tokenized_dataset_eval

        # barrier after data processing 
        dist.barrier(device_ids=[local_rank])

        # load after processed 
        loaded_tokenized_dataset_train = datasets.load_from_disk(data_path + "/train")
        loaded_tokenized_dataset_eval = datasets.load_from_disk(data_path + "/eval")

        loaded_tokenized_dataset_train.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        loaded_tokenized_dataset_eval.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    ###########################
    ####### Data Loader + Sampler + Optimizer  
    ###########################

    # Using distributed sampler and setting up our data loader 
    sampler = DistributedSampler(loaded_tokenized_dataset_train, num_replicas=world_size, rank=rank, drop_last=True)

    # Shuffle = False in dataloader when using a sampler
    train_dataloader = DataLoader(loaded_tokenized_dataset_train, batch_size=batch_size, sampler=sampler,
                                  shuffle=False, num_workers=0, drop_last=True, pin_memory=False)
    

    # Creating sampler + dataloader for eval too
    sampler_eval = sampler = DistributedSampler(loaded_tokenized_dataset_eval, num_replicas=world_size, rank=rank, drop_last=True)
    eval_dataloader = DataLoader(loaded_tokenized_dataset_eval, batch_size=batch_size, sampler=sampler,
                                  shuffle=False, num_workers=0, drop_last=True, pin_memory=False)
    
    # Calc training steps 
    num_training_steps = len(train_dataloader) * num_epochs

    warmup_steps = int(0.1 * num_training_steps)  # 10% warmup

    # Optimizer + scheduler 
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)  # Example LR and weight decay
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
    )
    
    ###########################
    ####### CHECKPOINTING 
    ###########################

    # Load checkpoint based on env var 
    if reload_checkpoint is not None:
        logging.info(f"User requested to reload checkpoint: {reload_checkpoint}, loading checkpoints")

        # State needs to look the same as the one that was saved
        state_to_load = {
            'model': model,
            'optimizer': optimizer,
            'scheduler': scheduler.state_dict(), 
            'epoch': -1,
        }
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            # dc.load automatically loads the sharded states into the provided model/optimizer
            dc.load(
                state_dict=state_to_load,
                checkpoint_id=ckpt_path + f'/{reload_checkpoint}' # Directory to load from
            )

        # Now update your training state from the loaded dict
        scheduler.load_state_dict(state_to_load['scheduler'])
        epoch = state_to_load['epoch'] + 1

    # else init directory and start from scratch 
    else:
        epoch = 0
        # Create checkpoint 
        if dist.get_rank() == 0:
            os.makedirs(ckpt_path, exist_ok=True)
        dist.barrier(device_ids=[local_rank])


    ###########################
    ####### TRAIN STEP FUNCTION
    ###########################
    def train(model, rank, train_dataloader, optimizer, scheduler):

        model.train() # Set model to train mode 
        train_loss = 0
 
        for batch_idx, batch in enumerate(train_dataloader):
            batch_start_time = time.time()

            # Put data on the devices
            batch = {k: v.to(rank) for k, v in batch.items()}
            input_ids = batch['input_ids']

            # Shift labels for next token prediction and then use pad token to replace where shifted
            labels = input_ids[:, 1:].clone()
            labels = F.pad(labels, (0, 1), mode='constant', value=tokenizer.pad_token_id)
            labels[labels == tokenizer.pad_token_id] = -100

            # Input data to model 
            out = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=labels)
            loss = out.loss
            loss = loss / accumulation_steps
            train_loss += float(loss)
            loss.backward()

            # Gradient accumulation if needed 
            if ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == len(train_dataloader)):
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            # Some logging based on batch log interval 
            if rank == 0 and batch_idx % batch_log_interval == 0:
                logging.info(f"Training Batch {batch_idx}/{len(train_dataloader)} Loss: {loss.item():.4f} Time: {time.time() - batch_start_time:.2f}s")


        # Avg the loss across the batches 
        train_loss /= len(train_dataloader)

        # Convert train_loss to a tensor before all_reduce
        train_loss = torch.tensor(train_loss).to(rank)

        dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)  # Sum losses across all processes
        train_loss /= dist.get_world_size()  # Average the summed loss
        if rank == 0:
            logging.info('Train set: Average loss for epoch: {:.4f}'.format(train_loss.item())) #.item() for logging


    ###########################
    ####### EVAL STEP FUNCTION
    ###########################
    def evaluate(model, rank, eval_dataloader):
    
        model.eval()  # Set the model to evaluation mode
        total_eval_loss = 0.0
        num_eval_batches = 0

        with torch.no_grad():  # IMPORTANT: Disable gradient calculations
            for batch_idx, batch in enumerate(eval_dataloader):
                batch_start_time = time.time()

                batch = {k: v.to(rank) for k, v in batch.items()}
                
                # Move batch to the correct device for the current FSDP rank
                input_ids = batch['input_ids']

                # Prepare labels: shift input_ids for next token prediction
                # This is the same as your training label preparation
                labels = input_ids[:, 1:].clone()
                labels = F.pad(labels, (0, 1), mode='constant', value=tokenizer.pad_token_id)
                labels[labels == tokenizer.pad_token_id] = -100  # Ignore pad tokens in loss

                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=batch['attention_mask'], labels=labels)
                loss = outputs.loss

                total_eval_loss += loss.item()  
                num_eval_batches += 1

                if rank == 0 and batch_idx % batch_log_interval == 0:
                    logging.info(f"Eval Batch {batch_idx}/{len(eval_dataloader)} Loss: {loss.item():.4f} Time: {time.time() - batch_start_time:.2f}s")


        # Calculate average loss for this rank
        avg_eval_loss_rank = total_eval_loss / num_eval_batches if num_eval_batches > 0 else 0.0

        # Convert to tensor for all_reduce
        eval_loss_tensor = torch.tensor(avg_eval_loss_rank, device=rank)

        # Reduce losses from all processes (sum and then average)
        dist.all_reduce(eval_loss_tensor, op=dist.ReduceOp.SUM)
        avg_eval_loss_epoch = eval_loss_tensor.item() / dist.get_world_size()

        if rank == 0:
            logging.info(f'Eval set: Average loss for epoch: {avg_eval_loss_epoch:.4f}')

    ###########################
    ####### Training Loop
    ###########################
    torch.cuda.empty_cache()
    if rank == 0:
        logging.info(f"Starting training from epoch: {epoch + 1}")

    for i in range(epoch, num_epochs):

        # set epoch on the sampler 
        train_dataloader.sampler.set_epoch(i)

        if rank == 0:
            logging.info(f"Epoch {i}")
        
        # Time measurements
        start = time.time()

        # Train function 
        train(model, local_rank, train_dataloader, optimizer, scheduler)
        if rank == 0:
            logging.info(f"Epoch training time: {time.time() - start}")

        # Eval function
        evaluate(model, local_rank, eval_dataloader)

        # Check checkpoint epochs against current epoch and save a checkpoint if needed
        if i % checkpoint_epochs == 0 and i != 0:
            state_to_save = {
                'model': model,  # Pass the FSDP model directly
                'optimizer': optimizer, # Pass the optimizer directly
                'scheduler': scheduler.state_dict(),
                'epoch': i,
            }
            with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
                # dc.save handles coordinating saves across ranks.
                dc.save(
                    state_dict=state_to_save,
                    checkpoint_id=ckpt_path + f"/{epoch}" # Use directory name for id
                )

    dist.destroy_process_group()


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 1, f"Requires at least 1 GPUs to run, but got {n_gpus}"
    train_fn()