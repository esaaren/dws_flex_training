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
from torch.distributed.fsdp.wrap import (_or_policy, #
                                         lambda_auto_wrap_policy,
                                         transformer_auto_wrap_policy)
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload 

from peft import get_peft_model, LoraConfig, TaskType
from tqdm.auto import tqdm
import transformers
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

# data stuff 
dataset_name=os.environ.get('DATASET_NAME', "dair-ai/emotion")
data_path = os.environ.get('DATA_PATH', '/home/esaarenvirta/training_data')
ckpt_path = os.environ.get('CHECKPOINT_PATH', '/home/esaarenvirta/checkpoints')
data_split_train=os.environ.get('DATA_SPLIT_TRAIN', 'train')
data_split_eval=os.environ.get('DATA_SPLIT_EVAL', 'validation')
data_split_test=os.environ.get('DATA_SPLIT_EVAL', 'test')
data_subset=os.environ.get('DATA_SUBSET', 'split')
dataset_name=os.environ.get('DATASET_NAME', "dair-ai/emotion")

# tokenizer stuff 
tokenizer_batch_size=int(os.environ.get('TOKENIZER_BATCH_SIZE', 10000))
max_length=int(os.environ.get('SEQUENCE_LENGTH', 325)) # Max size of the dataset we're using is ~300 so we can add some overhead for our formatting
tokenizer_num_proc=int(os.environ.get('TOKENIZER_NUM_PROC', 100))

# model stuff 
batch_size=int(os.environ.get('BATCH_SIZE', 8))
num_epochs=int(os.environ.get('EPOCHS', 5)) # 3 is a good amount of fine tuning, more epochs has more learning, higher hallucinations/overfitting 
accumulation_steps=int(os.environ.get('GRAD_ACC_STEPS', 1))
model_name=os.environ.get('MODEL_NAME', "meta-llama/Llama-3.1-8B")

warmup = float(os.environ.get('WARMUP', 0.1))
learning_rate = float(os.environ.get('LEARNING_RATE', 0.00002)) # Higher for faster training, lower for more stable but more epochs, set between 1e-4 and 5e-5 
weight_decay= float(os.environ.get('WEIGHT_DECAY', 0.01))

# peft stuff 
peft=os.environ.get('USE_PEFT', 'False')
if peft == 'False':
    peft = False
else:
    peft = True
grad_checkpointing = os.environ.get('GRAD_CHECKPOINTING_ENABLE', 'False')
peft_r = int(os.environ.get('PEFT_R', 16))
peft_alpha = int(os.environ.get('PEFT_ALPHA', 32))
peft_dropout= float(os.environ.get('PEFT_DROPOUT', 0.05))

# checkpoint stuff 
checkpoint_epochs=int(os.environ.get('CHECKPOINT_EPOCHS', 1)) # how many epochs should we checkpoint between, 1 will checkpoint at every epoch
reload_checkpoint=os.environ.get('RELOAD_CHECKPOINT', "None") # Needs to be a string of the folder under ckpt_path e.g "1" or "None"
if reload_checkpoint == "None":
    reload_checkpoint = None

# others 
nccl_timeout=int(os.environ.get('NCCL_TIMEOUT', 120))
batch_log_interval = int(os.environ.get('BATCH_LOG_INTERVAL', 100))
test_after_training = os.environ.get('TEST_AFTER_TRAINING', 'True')



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

    # Add barrier timeout for when we're processing large pieces of data (or downloading checkpoints)
    dist.init_process_group("nccl", timeout=desired_timeout) 

    rank = dist.get_rank()

    logging.info(f"Running DDP on rank {rank}, local rank {local_rank} with world size {world_size}")
    torch.cuda.set_device(local_rank)
    logging.info(f"Inside rank  {rank} {local_rank}")

    ###########################
    ####### Model + Tokenizer Download 
    ###########################
    # avoid deadlocks when all ranks download model shards 
    if rank == 0:
        logging.info(f"Rank {rank} downloading model and config...")

        # Tokenizer and model init on rank 0 so no deadlocks when downloading to GCS
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
    tokenizer.padding_side = 'left'

    # Tokenizer and text processing functions 
    # Pre-process the data so we create an instruction prompt
    def add_prompt(text):
        prompt = f"Classify the emotion of this text with a single word only: {text}, Emotion: "
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
            task_type=TaskType.CAUSAL_LM, inference_mode=False, r=peft_r, lora_alpha=peft_alpha, lora_dropout=peft_dropout
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
                 None, # TODO: Look into implementing
            sharding_strategy=ShardingStrategy.FULL_SHARD, # It shards the parameters, gradients, and optimizer states across all GPUs.
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True, # Reduces the number of all-gather operations, which can be expensive.
            sync_module_states=False, # This is important. Since we're loading the full model initially (on the CPU), we don't want FSDP to try to synchronize module states across processes at this stage. The synchronization will happen naturally during the first forward/backward pass.
            param_init_fn=( # This is set to None, but the sample template shows an example of how you could use a custom initialization function. We're using on the pretrained weights, so we don't need a custom initialization.
                (
                    lambda module: module.to_empty(
                        device=torch.device("cuda"), recurse=False
                    )
                )
                if False and rank != 0 # Set to true if you want to save cpu memory by loading pretrained model on rank0 only
                else None
            ),
            use_orig_params=True, # Keeps a reference to the original parameters.
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
            
            dataset_train = load_dataset(dataset_name, data_subset, split=data_split_train)
            dataset_eval = load_dataset(dataset_name, data_subset, split=data_split_eval)
            df_train = pd.DataFrame(dataset_train)
            df_eval = pd.DataFrame(dataset_eval)

            # map the labels TODO: Make this templateable 
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

    eval_dataloader = DataLoader(loaded_tokenized_dataset_eval, batch_size=batch_size, sampler=sampler_eval,
                                  shuffle=False, num_workers=0, drop_last=True, pin_memory=False)
    
    # Calc training steps 
    num_training_steps = len(train_dataloader) * num_epochs

    warmup_steps = int(warmup * num_training_steps)  

    # Optimizer + scheduler 
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay) 
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

            # Input data into the model
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

                total_eval_loss += loss.item()  # item() gets the Python number
                num_eval_batches += 1

                if rank == 0 and batch_idx % batch_log_interval == 0:
                    logging.info(f"Eval Batch {batch_idx}/{len(eval_dataloader)} Loss: {loss.item():.4f} Time: {time.time() - batch_start_time:.2f}s")
                # if rank == 0: eval_bar.update(1); eval_bar.set_postfix(loss=loss.item())


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

        # Important to set epoch on data loader sampler #TODO: Does it matter for eval dataloader? 
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
        # TODO: Can make a better checkpoint manager  
        if i % checkpoint_epochs == 0 and i != 0:
            state_to_save = {
                'model': model,  # Pass the FSDP model directly
                'optimizer': optimizer, # Pass the optimizer directly
                'scheduler': scheduler.state_dict(),
                'epoch': i,
            }
            
            # dc.save handles coordinating saves across ranks.
            dc.save(
                state_dict=state_to_save,
                checkpoint_id=ckpt_path + f"/{i}" # Use directory name for id
            )


    logging.info("Running final checkpoint against testing data")



    ###########################
    ####### Evaluate against the test data 
    ###########################

    if test_after_training == 'True':

        # Process data like we did with the other datasets
        if rank == 0:
            dataset_test = load_dataset(dataset_name, data_subset, split=data_split_test)
            df_test = pd.DataFrame(dataset_test)

            label_mapping = {
                0: 'sadness',
                1: 'joy',
                2: 'love',
                3: 'anger',
                4: 'fear',
                5: 'surprise'
            }
            
            # Format the prompt and then apply the str version of the labels 
            df_test['text'] = df_test['text'].apply(add_prompt)
            df_test['label_text'] = df_test['label'].map(label_mapping)

            dataset_test = Dataset.from_pandas(df_test)

            print(dataset_test[0]['text']) # quick sanity check on formatting 
            print(dataset_test[0]['label_text']) # quick sanity check on formatting  


            tokenized_dataset_test = dataset_test.map(process_text, batch_size=tokenizer_batch_size, num_proc=tokenizer_num_proc, batched=True)
            tokenized_dataset_test.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label_text']) # Not setting data type to torch is common pitfall
            tokenized_dataset_test.save_to_disk(data_path + '/test')

            del tokenized_dataset_test

        dist.barrier(device_ids=[local_rank])

        # Load data and sampler/dataloader for the test data
        loaded_tokenized_dataset_test = datasets.load_from_disk(data_path + "/test")
        sampler_test = sampler = DistributedSampler(loaded_tokenized_dataset_test, num_replicas=world_size, rank=rank, drop_last=True)
        test_dataloader = DataLoader(loaded_tokenized_dataset_test, batch_size=batch_size, sampler=sampler_test,
                                  shuffle=False, num_workers=0, drop_last=True, pin_memory=False)
    
        # Can't forget to set eval here 
        model.eval()

        correct_predictions = 0
        total_samples = 0

        for batch_idx, batch in enumerate(test_dataloader):

            input_ids = batch['input_ids'].to(local_rank)
            attention_mask = batch['attention_mask'].to(local_rank)
            ground_truth_labels_str_list = batch['label_text']

            generated_outputs = model.generate(
                 input_ids=input_ids, 
                 attention_mask=attention_mask,
                 max_new_tokens=5,  # Adjust based on expected label length (e.g., "surprise" is 8 chars)
                 pad_token_id=tokenizer.pad_token_id,
                 eos_token_id=tokenizer.eos_token_id,
                 do_sample=False  # Use greedy decoding for deterministic output
             )
            
            # The generated_outputs contain the prompt + generated tokens.
            # We need to slice off the prompt part.
            num_prompt_tokens = batch['input_ids'].shape[1]
            generated_text_ids = generated_outputs[:, num_prompt_tokens:]

            # Decode the generated tokens
            decoded_predictions_raw = tokenizer.batch_decode(generated_text_ids, skip_special_tokens=True)

            for i in range(len(decoded_predictions_raw)):
                raw_prediction = decoded_predictions_raw[i]
                true_label = ground_truth_labels_str_list[i].strip().lower()

                # Refined extraction of the predicted emotion (first word, cleaned)
                predicted_emotion_word = ""
                raw_prediction_stripped = raw_prediction.strip()
                if raw_prediction_stripped: # If not an empty string after stripping
                    # Take the first "word"
                    predicted_emotion_word = raw_prediction_stripped.lower().split(' ')[0]
                    # Remove common trailing punctuation from the extracted word
                    predicted_emotion_word = predicted_emotion_word.rstrip(",.!?:;")

                # Sanity check on tuned labels 
                if rank == 0 and batch_idx < 1 and i < 3:
                    original_prompt_decoded = tokenizer.decode(input_ids[i], skip_special_tokens=True)
                    if tokenizer.eos_token: # Clean up prompt for logging if EOS is pad
                            original_prompt_decoded = original_prompt_decoded.split(tokenizer.eos_token)[0]
                    #logging.info(f"  Prompt: '{original_prompt_decoded}'")
                    #logging.info(f"  True Label: '{true_label}'")
                    #logging.info(f"  Generated (raw full): '{raw_prediction}'")
                    #logging.info(f"  Extracted Predicted Word: '{predicted_emotion_word}'")

                # Compare the extracted word with the true label
                if true_label == predicted_emotion_word:
                    correct_predictions += 1
         

            total_samples += len(ground_truth_labels_str_list)

        if rank == 0 and (batch_idx + 1) % batch_log_interval == 0:
             temp_acc = (correct_predictions / total_samples) if total_samples > 0 else 0
             logging.info(f"Test Accuracy Batch {batch_idx+1}/{len(test_dataloader)}: Temp Accuracy (rank 0): {temp_acc:.4f}")

        # Aggregate results from all processes
        correct_predictions_tensor = torch.tensor(correct_predictions, dtype=torch.long).to(local_rank)
        total_samples_tensor = torch.tensor(total_samples, dtype=torch.long).to(local_rank)

        dist.all_reduce(correct_predictions_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)

        final_accuracy = 0.0
        total_samples_agg = total_samples_tensor.item()
        if total_samples_agg > 0:
            final_accuracy = correct_predictions_tensor.item() / total_samples_agg

        if rank == 0:
            logging.info(f"--- Test Set Accuracy Results ---")
            logging.info(f"Total Correct Predictions (aggregated): {correct_predictions_tensor.item()}")
            logging.info(f"Total Samples (aggregated): {total_samples_agg}")
            logging.info(f"Final Test Accuracy: {final_accuracy:.4f}")


    # Destroy the process group and end the run
    dist.destroy_process_group()


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 1, f"Requires at least 1 GPUs to run, but got {n_gpus}"
    train_fn()