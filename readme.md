# Cluster setups for different GCP accelerators with sample torch jobs

This guide is meant to be a basic instruction on how to set up a GKE training cluster, run a quick network test, run a quick pytorch test and demonstrate how to submit a distributed torch training job. 

## Inspired by:
```
https://cloud.google.com/kubernetes-engine/docs/how-to/gpu-bandwidth-gpudirect-tcpx
```
for a3-mega 

### And:
```
https://cloud.google.com/ai-hypercomputer/docs/create/gke-ai-hypercompute-custom
```
for a3-ultra 

### Retrofitted to support:  ```DWS Flex```

### Navigate to the individual directories:
```a3-mega``` or ```a3-ultra``` for cluster setup instructions

### Navigate to ```kubernetes``` for sample kubernetes jobs for distributed torch 

### Navigate to ```torch``` for a sample torch job for testing the infrastructure