# Check valid GKE versions
```
gcloud container get-server-config --format="yaml(validMasterVersions)" --zone=${ZONE} --project=${PROJECT}
```

# Exports with sample presets
```
export PREFIX="erika3-mega-us-east4"
export REGION="us-east4"	
export PROJECT=""	
export ZONE="us-east4-b"
export GKE_VERSION=1.31.7-gke.1265000
export CLUSTER_NAME="a3-mega-erik"
export GSBUCKET='erik-mega-training'	
export PROJECT_NUMBER=$(gcloud projects describe ${PROJECT} --format="value(projectNumber)") && \
export NAMESPACE='default'
export KSA_NAME='erik-ksa'
export HF_TOKEN=
```



# Networking setup 
```
for N in $(seq 1 8); do
gcloud compute networks create ${PREFIX}-net-$N \
    --subnet-mode=custom \
    --mtu=8244

gcloud compute networks subnets create ${PREFIX}-sub-$N \
    --network=${PREFIX}-net-$N \
    --region=us-east4 \
    --range=192.168.$N.0/24

gcloud compute firewall-rules create ${PREFIX}-internal-$N \
  --network=${PREFIX}-net-$N \
  --action=ALLOW \
  --rules=tcp:0-65535,udp:0-65535,icmp \
  --source-ranges=192.168.0.0/16
done
```

# Cluster setup 
```
gcloud --project ${PROJECT} beta container clusters create ${CLUSTER_NAME} \
    --enable-dataplane-v2 \
    --enable-ip-alias \
    --region ${REGION} \
    --node-locations ${ZONE} \
    --enable-multi-networking \
    --cluster-version ${GKE_VERSION} \
    --no-enable-autoupgrade \
    --workload-pool=${PROJECT}.svc.id.goog \
    --addons GcsFuseCsiDriver
```

# Cluster get auth
```
gcloud container clusters get-credentials ${CLUSTER_NAME} --region ${REGION} --project ${PROJECT}
```

# Create the network objects (you need to edit this one, check where the networking prefixes are and replace with yours)
```
kubectl apply -f network.yaml
```

# Nodepool setup 
# NOTE: CHECK DRIVERS https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded-latest.yaml
# On GKE 1.32 as of Apr 28 2025 LATEST will get you driver 570 which is CUDA 12.8, DEFAULT gives you 535 which is 12.2 
```
gcloud beta container node-pools create a3mega-multi-nic-dws \
    --region ${REGION} \
    --cluster=${CLUSTER_NAME} \
    --project=${PROJECT}\
    --node-locations=${ZONE}\
    --no-enable-autoupgrade \
    --no-enable-autorepair\
    --enable-queued-provisioning \
    --reservation-affinity=none \
    --location-policy=ANY \
    --enable-autoscaling \
    --total-max-nodes=10 \
    --num-nodes=0 \
    --accelerator=type=nvidia-h100-mega-80gb,count=8,gpu-driver-version=LATEST \
    --machine-type=a3-megagpu-8g \
    --additional-node-network network=${PREFIX}-net-1,subnetwork=${PREFIX}-sub-1 \
    --additional-node-network network=${PREFIX}-net-2,subnetwork=${PREFIX}-sub-2 \
    --additional-node-network network=${PREFIX}-net-3,subnetwork=${PREFIX}-sub-3 \
    --additional-node-network network=${PREFIX}-net-4,subnetwork=${PREFIX}-sub-4 \
    --additional-node-network network=${PREFIX}-net-5,subnetwork=${PREFIX}-sub-5 \
    --additional-node-network network=${PREFIX}-net-6,subnetwork=${PREFIX}-sub-6 \
    --additional-node-network network=${PREFIX}-net-7,subnetwork=${PREFIX}-sub-7 \
    --additional-node-network network=${PREFIX}-net-8,subnetwork=${PREFIX}-sub-8 \
    --enable-gvnic \
    --no-enable-autoupgrade \
    --scopes "https://www.googleapis.com/auth/cloud-platform"
```

# Add secret 
```
kubectl create secret generic hf-secret \
--from-literal=hf_api_token=${HF_TOKEN} 
```

# Create GCS bucket 
```
gcloud storage buckets create gs://${GSBUCKET} \
    --uniform-bucket-level-access \
    --location=${REGION}\
    --enable-hierarchical-namespace
```

# Create GKE SA 
```
kubectl create serviceaccount ${KSA_NAME}
```

# Add perms to GCS bucket for GCS fuse
```
gcloud storage buckets add-iam-policy-binding gs://${GSBUCKET} \
  --member "principal://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${PROJECT}.svc.id.goog/subject/ns/${NAMESPACE}/sa/${KSA_NAME}" \
  --role "roles/storage.objectUser"
```

# NCCL plugin
# NCCL 2.23.4
```
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/gpudirect-tcpxo/nccl-tcpxo-installer.yaml
kubectl get pods -n=kube-system -l=name=nccl-tcpxo-installer
```
# NRI Device injector plugin
```
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nri_device_injector/nri-device-injector.yaml
kubectl get pods -n=kube-system -l=name=device-injector
```

# Create the provisioning request
```
kubectl apply -f p_request.yaml
kubectl describe provreq provreq-gpu
```

# General check kube system
```
kubectl get pod -n kube-system
```

# NCCL test (2 host and 4 host) - Wait a min or so after NCCL deployments are up for the injectors + NCCL plugins to install
```
kubectl apply -f nccl-latest.yaml
kubectl exec --stdin --tty --container=nccl-test nccl-test-host-1 -- /scripts/allgather.sh nccl-host-1 nccl-host-2
kubectl exec --stdin --tty --container=nccl-test nccl-test-host-1 -- /scripts/allgather.sh nccl-host-1 nccl-host-2 nccl-host-3 nccl-host-4
```

# Docker helpers

# Auth 
```
gcloud auth configure-docker ${REGION}-docker.pkg.dev
```

# Create repos 
```
gcloud artifacts repositories create erik-torch-images --repository-format=docker --location=${REGION} --project=${PROJECT}
```

# Torch (In the torch dir.)
```
docker build -f Dockerfile.mega -t ${REGION}-docker.pkg.dev/${PROJECT}/erik-torch-images/torch-mega-job:latest .
docker push ${REGION}-docker.pkg.dev/${PROJECT}/erik-torch-images/torch-mega-job:latest 
```

# Watch torch job
```
kubectl apply -f ../kubernetes/mega_torch_job.yaml
kubectl logs -l app=torch-job -c job -f
watch -n 1 kubectl logs -l app=torch-job -c job -f
```