# Simple distributed torch examples
### Docker build instructions can be found in the folders for each hardware type

Use:
```
torch_allgather.py
``` 

For a simple smoke test after cluster setup

Use:
```
fsdp.py
```

For a simple torch training recipe


Edit PYTORCH_SCRIPT_NAME in:
```/kubernetes/<mega_or_ultra>_torch_job.yaml```
to switch between the two scripts

### We can measure the improvement difference on epoch time between infrastructure, for ex: 
#### Single node H100 training: 
```154s``` 
per epoch at batch size == 8 

#### 4 node H200 training: 
```16s``` 
per epoch at batch size == 32 