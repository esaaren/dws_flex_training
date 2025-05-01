# Simple distributed torch examples
### Docker build instructions can be found in the folders for each hardware type

Use:
```
torch_allgather.py
``` 

For a simple smoke test after cluster setup to make sure that NCCL is working with torch. 

Use:
```
fsdp.py
```

For a simple torch training recipe

There is also a ```local_launch.sh``` script if you want to run this repo on a single node locally (like an H100 or H200) but care at the default environment vars in fsdp.py 

Edit PYTORCH_SCRIPT_NAME in:
```/kubernetes/<mega_or_ultra>_torch_job.yaml```
to switch between the two scripts

### We can measure the improvement difference on epoch time between infrastructure, for ex: 
#### Single node H100 training: 
```154s``` 
per epoch at batch size == 8 

### 4 node H100 mega training:
```38s```
per epoch at batch size == 16 

#### 4 node H200 training (we can fit a larger batch size): 
```16s``` 
per epoch at batch size == 32 


### A sample output after training for 5 epochs with lr==0.00002 and batch==32 on A3-ultra: 
```
INFO:root:Train set: Average loss for epoch: 2.7037
INFO:root:Eval set: Average loss for epoch: 2.6999

INFO:root:--- Test Set Accuracy Results ---
INFO:root:Total Correct Predictions (aggregated): 576
INFO:root:Total Samples (aggregated): 1024
INFO:root:Final Test Accuracy: 0.5625
```