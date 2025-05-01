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

import torch
import torch.distributed as dist
import os
import time

def run_allgather_test():
    """Runs a simple distributed all-gather test on GPU."""

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)  # Set the current CUDA device
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()

    print(f"Rank {rank}/{world_size} (local rank {local_rank}): Initialized distributed environment.")

    # Create some local data on each process and move it to the GPU
    local_data = torch.tensor([rank + 1] * 3).cuda(local_rank)
    print(f"Rank {rank}: Local data on GPU = {local_data}")

    # Create a list to hold the gathered data from all processes (on GPU)
    gather_list = [torch.empty_like(local_data) for _ in range(world_size)]

    # Perform the all-gather operation
    start_time = time.time()
    dist.all_gather(gather_list, local_data)
    end_time = time.time()

    print(f"Rank {rank}: Gathered data on GPU = {gather_list}")
    print(f"Rank {rank}: All-gather took {end_time - start_time:.4f} seconds.")

    # Verify the gathered data
    expected_data = [torch.tensor([i + 1] * 3).cuda(local_rank) for i in range(world_size)]
    if all(torch.equal(g, e) for g, e in zip(gather_list, expected_data)):
        print(f"Rank {rank}: All-gather test passed on GPU!")
    else:
        print(f"Rank {rank}: All-gather test FAILED on GPU!")

    dist.destroy_process_group()

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is not available. Please ensure you have a CUDA-enabled GPU and the correct drivers installed.")
    else:
        run_allgather_test()