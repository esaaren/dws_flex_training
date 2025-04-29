# Exports with sample presets
```
export REGION="us-central1"
export ZONE="us-central1-b"
export PROJECT=""
export GVNIC_NETWORK_PREFIX="erik-ultra-gvnic"
export RDMA_NETWORK_PREFIX="erik-ultra-rdma"
export KSA_NAME=erik-ksa
export GSBUCKET=erik-torch-training
export PROJECT_NUMBER=$(gcloud projects describe ${PROJECT} --format="value(projectNumber)") && \
export NAMESPACE='default'
export CLUSTER_NAME="erik-ultra-dws"
export HF_TOKEN=
export GKE_VERSION=1.32.3-gke.1717000
```

# Create a VPC for the additional Google Titanium CPU NIC
```
gcloud compute --project=${PROJECT?} \
  networks create \
  ${GVNIC_NETWORK_PREFIX?}-net \
  --subnet-mode=custom

gcloud compute --project=${PROJECT?} \
  networks subnets create \
  ${GVNIC_NETWORK_PREFIX?}-sub \
  --network=${GVNIC_NETWORK_PREFIX?}-net \
  --region=${REGION?} \
  --range=192.168.0.0/24

gcloud compute --project=${PROJECT?} \
  firewall-rules create \
  ${GVNIC_NETWORK_PREFIX?}-internal \
  --network=${GVNIC_NETWORK_PREFIX?}-net \
  --action=ALLOW \
  --rules=tcp:0-65535,udp:0-65535,icmp \
  --source-ranges=192.168.0.0/16
```

# Create HPC VPC for the RDMA NICs with 8 subnets.
```
gcloud beta compute --project=${PROJECT?} \
  networks create ${RDMA_NETWORK_PREFIX?}-net \
  --network-profile=${ZONE?}-vpc-roce \
  --subnet-mode=custom
```

# Create subnets for the HPC VPC.
```
for N in $(seq 0 7); do
  gcloud compute --project=${PROJECT?} \
    networks subnets create \
    ${RDMA_NETWORK_PREFIX?}-sub-$N \
    --network=${RDMA_NETWORK_PREFIX?}-net \
    --region=${REGION?} \
    --range=192.168.$((N+1)).0/24 &  # offset to avoid overlap with gvnics
done
```


# Create a cluster 
```
gcloud container clusters create ${CLUSTER_NAME} \
    --region=${REGION}$ \
    --cluster-version=${CLUSTER_VERSION}$ \
    --workload-pool=${PROJECT}$.svc.id.goog \
    --services-ipv4-cidr=10.65.0.0/19 \
    --cluster-ipv4-cidr=10.64.0.0/19 \
    --enable-dataplane-v2 \
    --enable-ip-alias \
    --enable-multi-networking \
    --no-enable-autoupgrade \
    --addons=GcsFuseCsiDriver
```

# Create the network objects (you need to edit this one, check where the networking prefixes are and replace with yours)
```
kubectl apply -f network.yaml
```

# Get cluster creds
```
gcloud container clusters get-credentials ${CLUSTER_NAME} --region ${REGION} --project ${PROJECT}
```

# Create a nodepool
```
gcloud container node-pools create ultra-nodepool-dws \
    --cluster=${CLUSTER_NAME}$ \
    --region=${REGION}$ \
    --node-locations=${ZONE}$ \
    --machine-type=a3-ultragpu-8g \
    --accelerator=type=nvidia-h200-141gb,count=8,gpu-driver-version=DEFAULT \
    --scopes="https://www.googleapis.com/auth/cloud-platform" \
    --reservation-affinity=none \
    --location-policy=ANY \
    --enable-queued-provisioning \
    --flex-start \
    --no-enable-autoupgrade \
    --no-enable-autorepair \
    --enable-autoscaling \
    --num-nodes=0 \
    --total-max-nodes=10 \
    --additional-node-network=network=${GVNIC_NETWORK_PREFIX}-net,subnetwork=${GVNIC_NETWORK_PREFIX}-sub \
    --additional-node-network=network=${RDMA_NETWORK_PREFIX}-net,subnetwork=${RDMA_NETWORK_PREFIX}-sub-0 \
    --additional-node-network=network=${RDMA_NETWORK_PREFIX}-net,subnetwork=${RDMA_NETWORK_PREFIX}-sub-1 \
    --additional-node-network=network=${RDMA_NETWORK_PREFIX}-net,subnetwork=${RDMA_NETWORK_PREFIX}-sub-2 \
    --additional-node-network=network=${RDMA_NETWORK_PREFIX}-net,subnetwork=${RDMA_NETWORK_PREFIX}-sub-3 \
    --additional-node-network=network=${RDMA_NETWORK_PREFIX}-net,subnetwork=${RDMA_NETWORK_PREFIX}-sub-4 \
    --additional-node-network=network=${RDMA_NETWORK_PREFIX}-net,subnetwork=${RDMA_NETWORK_PREFIX}-sub-5 \
    --additional-node-network=network=${RDMA_NETWORK_PREFIX}-net,subnetwork=${RDMA_NETWORK_PREFIX}-sub-6 \
    --additional-node-network=network=${RDMA_NETWORK_PREFIX}-net,subnetwork=${RDMA_NETWORK_PREFIX}-sub-7
```

# Create the SA and policy for the GCS bucket we use 
```
kubectl create serviceaccount ${KSA_NAME}
```

# Add perms to GCS bucket for GCS fuse
```
gcloud storage buckets add-iam-policy-binding gs://${GSBUCKET} \
  --member "principal://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${PROJECT}.svc.id.goog/subject/ns/${NAMESPACE}/sa/${KSA_NAME}" \
  --role "roles/storage.objectUser"
```

# NCCL RDMA installer (April 24 2025 this installs NCCL 2.25.1)
```
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/refs/heads/master/gpudirect-rdma/nccl-rdma-installer.yaml
kubectl get pod -n kube-system | grep rdma
```

# Create the provisioning request
```
kubectl apply -f p_request.yaml
kubectl describe provreq 
```

# Quick NCCL test to verify 
```
kubectl apply -f nccl.yaml 
kubectl exec nccl-test-host-1 -it -- /usr/local/gib/scripts/run_nccl_tests.sh -t all_gather -b 1K -e 8G nccl-host-1 nccl-host-2
```

# Torch Image building (In the torch dir.)
```
gcloud auth configure-docker ${REGION}-docker.pkg.dev
gcloud artifacts repositories create erik-torch-images --repository-format=docker --location=${REGION} --project=${PROJECT}
docker build -f Dockerfile.ultra -t ${REGION}-docker.pkg.dev/${PROJECT}/erik-torch-images/torch-ultra-job:latest .
docker push ${REGION}-docker.pkg.dev/${PROJECT}/erik-torch-images/torch-ultra-job:latest 
```

# Run the torch job 
```
kubectl apply -f ../kubernetes/ultra_torch_job.yaml 
kubectl logs -l app=torch-job -c job -f
watch -n 1 kubectl logs -l app=torch-job -c job -f
```