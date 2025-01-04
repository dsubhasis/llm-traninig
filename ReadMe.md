# ML Model Fine-tuning Deployment Guide

This guide explains how to deploy your ML model fine-tuning workload using Docker and Kubernetes.

## Prerequisites

- Docker installed and configured
- Kubernetes cluster set up and running
- kubectl CLI tool installed
- Access to a container registry
- NVIDIA GPU drivers and runtime installed (for GPU support)

## Project Structure

```
project/
├── Dockerfile
├── kubernetes/
│   ├── deployment.yaml
│   └── configmap.yaml
├── .env
├── requirements.txt
└── finetune.py
```

## Step-by-Step Deployment Guide

### 1. Environment Setup

Create a `.env` file in your project root:

```env
WANDB_API_KEY=your_wandb_api_key
MODEL_PATH=/app/models
DATA_PATH=/app/data
```

### 2. Docker Image Build and Push

```bash
# Build the Docker image
docker build -t your-registry/finetune:latest .

# Log in to your container registry
docker login your-registry

# Push the image
docker push your-registry/finetune:latest
```

### 3. Kubernetes Deployment

#### 3.1 Create ConfigMap from .env file

```bash
# Create ConfigMap from .env file
kubectl create configmap env-config --from-file=.env=.env
```

#### 3.2 Apply Kubernetes manifests

```bash
# Apply the deployment configuration
kubectl apply -f kubernetes/deployment.yaml
```

### 4. Verify Deployment

```bash
# Check deployment status
kubectl get deployments

# Check pods status
kubectl get pods

# View pod logs
kubectl logs -f deployment/finetune-deployment
```

### 5. Monitor Training Progress

```bash
# Stream logs from the training pod
kubectl logs -f $(kubectl get pods -l app=finetune -o jsonpath='{.items[0].metadata.name}')
```

## Resource Management

### Storage

The deployment uses two Persistent Volume Claims:
- `model-storage-pvc`: For storing model checkpoints and artifacts
- `training-data-pvc`: For storing training data

Ensure you have sufficient storage provisioned in your cluster.

### GPU Resources

The deployment is configured to use 1 NVIDIA GPU. Modify the resource limits in `deployment.yaml` if you need different GPU configurations:

```yaml
resources:
  limits:
    nvidia.com/gpu: 1  # Modify as needed
```

## Troubleshooting

### Common Issues and Solutions

1. **Pod in Pending State**
   ```bash
   kubectl describe pod <pod-name>
   ```
   Common causes:
   - Insufficient resources
   - PVC not bound
   - GPU not available

2. **Container Startup Failure**
   ```bash
   kubectl logs <pod-name>
   ```
   Check for:
   - Environment variables
   - Mount point permissions
   - Python package conflicts

3. **GPU Not Detected**
   Verify GPU setup:
   ```bash
   kubectl exec -it <pod-name> -- nvidia-smi
   ```

## Cleaning Up

Remove deployment and resources:

```bash
# Delete deployment
kubectl delete -f kubernetes/deployment.yaml

# Delete ConfigMap
kubectl delete configmap env-config

# Delete PVCs (caution: this will delete persistent data)
kubectl delete pvc model-storage-pvc training-data-pvc
```

## Best Practices

1. **Version Control**
   - Tag Docker images with specific versions instead of using 'latest'
   - Version control your Kubernetes manifests

2. **Resource Requests**
   - Set appropriate CPU/Memory requests and limits
   - Monitor resource usage and adjust as needed

3. **Security**
   - Use private container registries
   - Regularly update base images and dependencies
   - Follow principle of least privilege

4. **Monitoring**
   - Set up monitoring for both Kubernetes resources and training metrics
   - Configure alerts for resource constraints and training failures

## Support

For additional support:
- Check Kubernetes documentation
- Review Docker documentation
- Consult ML framework-specific guides