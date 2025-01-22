# Slurm Plugin for AI/ML Integration with OCP and OCP AI

This document provides a detailed technical guide to implement a Slurm plugin that integrates AI/ML workloads with OpenShift Container Platform (OCP) and OpenShift AI. This guide includes step-by-step instructions and code snippets to help you build the plugin.

## Repository Structure
```plaintext
ocp-ai-slurm-plugin/
├── README.md
├── scripts/
│   ├── job_submit.lua
│   ├── ocp_integration.py
│   └── config.yaml
├── crds/
│   ├── mpi_job.yaml
│   └── tf_job.yaml
├── manifests/
│   ├── slurm-exporter.yaml
│   ├── namespace.yaml
│   └── storage-class.yaml
├── monitoring/
│   ├── grafana-dashboard.json
│   └── prometheus-config.yaml
└── Dockerfile
```

## Step-by-Step Implementation

### 1. **Set Up the Development Environment**
#### Prerequisites
- OpenShift CLI (`oc`)
- Python 3.x with `kubernetes` and `pyyaml` modules
- A running OCP cluster with NVIDIA GPU Operator and OpenShift AI enabled
- Slurm installed and configured

### 2. **Configure OpenShift Resources**
#### Create Namespace for Jobs
Create a namespace to isolate Slurm-managed workloads.
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: slurm-jobs
```
Apply the namespace:
```bash
oc apply -f manifests/namespace.yaml
```

#### Define Storage Class for Persistent Volumes
Use a CephFS-backed storage class:
```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: cephfs
provisioner: openshift-storage.cephfs.csi.ceph.com
parameters:
  clusterID: <cluster-id>
  pool: <pool-name>
```
Apply the storage class:
```bash
oc apply -f manifests/storage-class.yaml
```

### 3. **Develop the Slurm Plugin**
#### Create `job_submit.lua`
Modify Slurm’s job submission policy to delegate containerized workloads to OCP.
```lua
function slurm_job_submit(job_desc, part_list, submit_uid)
  if job_desc.gres == "gpu" then
    job_desc.script = "/scripts/ocp_integration.py"
  end
  return slurm.SUCCESS
end

slurm.log_info("Job submit plugin loaded.")
```
Place this script in Slurm’s `plugstack.conf.d` directory.

#### Python Integration Script (`ocp_integration.py`)
This script creates Kubernetes Jobs or CRDs for AI workloads.
```python
import yaml
from kubernetes import client, config

def create_k8s_job(job_name, image, namespace, gpu_count):
    config.load_kube_config()
    batch_v1 = client.BatchV1Api()

    job = client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=client.V1ObjectMeta(name=job_name),
        spec=client.V1JobSpec(
            template=client.V1PodTemplateSpec(
                spec=client.V1PodSpec(
                    containers=[client.V1Container(
                        name="ai-job",
                        image=image,
                        resources=client.V1ResourceRequirements(
                            limits={"nvidia.com/gpu": str(gpu_count)},
                            requests={"nvidia.com/gpu": str(gpu_count)}
                        )
                    )],
                    restart_policy="Never"
                )
            )
        )
    )

    batch_v1.create_namespaced_job(namespace=namespace, body=job)

if __name__ == "__main__":
    with open("config.yaml") as f:
        config_data = yaml.safe_load(f)

    create_k8s_job(
        job_name=config_data["job_name"],
        image=config_data["image"],
        namespace=config_data["namespace"],
        gpu_count=config_data["gpu_count"]
    )
```

#### Configuration File (`config.yaml`)
Define job parameters in a YAML file.
```yaml
job_name: ai-training-job
image: quay.io/myrepo/ml-image:latest
namespace: slurm-jobs
gpu_count: 2
```

### 4. **Enable Monitoring**
#### Prometheus Exporter for Slurm
Deploy a Prometheus exporter to collect metrics.
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: slurm-exporter
spec:
  replicas: 1
  selector:
    matchLabels:
      app: slurm-exporter
  template:
    metadata:
      labels:
        app: slurm-exporter
    spec:
      containers:
      - name: exporter
        image: prom/slurm-exporter:latest
        ports:
        - containerPort: 8080
```
Apply the deployment:
```bash
oc apply -f manifests/slurm-exporter.yaml
```

#### Grafana Dashboard
Import `monitoring/grafana-dashboard.json` into Grafana to visualize AI/ML job metrics.

### 5. **Build and Deploy the Plugin**
#### Dockerfile
Create a container image for the integration script.
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY scripts/ocp_integration.py .
COPY scripts/config.yaml .
RUN pip install kubernetes pyyaml
ENTRYPOINT ["python", "ocp_integration.py"]
```
Build and push the image:
```bash
docker build -t quay.io/myrepo/ocp-slurm-plugin:latest .
docker push quay.io/myrepo/ocp-slurm-plugin:latest
```

### 6. **Test the Integration**
1. Submit a GPU-based job:
   ```bash
   sbatch --gres=gpu:2 --job-name=ai-training-job script.sh
   ```
2. Verify the job is translated into an OCP workload:
   ```bash
   oc get pods -n slurm-jobs
   ```
3. Monitor metrics in Grafana.

### 7. **Future Enhancements**
- Add support for distributed training (MPIJob).
- Implement fault tolerance with checkpointing.
- Integrate with OpenShift Virtualization for hybrid workloads.

---
This implementation provides a solid foundation for enabling Slurm, OCP, and AI/ML workloads in a hybrid environment.