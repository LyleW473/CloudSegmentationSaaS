#!/bin/bash

# Install python-dotenv (For running the scripts)
pip install python-dotenv --no-cache-dir

# Run the scripts to create the configuration files
python scripts/create_cluster_config.py
python scripts/create_ingress_config.py
python scripts/create_deployment_config.py

# Set up the kind cluster
kind create cluster --name mycluster --config cluster-config-processed.yaml --wait 5m
kubectl cluster-info --context kind-mycluster
kubectl get nodes

# Build the Docker images
docker compose build

# Load individual Docker images into the kind cluster
kind load docker-image data-retrieval:latest --name mycluster
kind load docker-image data-processing:latest --name mycluster
kind load docker-image model-inference:latest --name mycluster
kind load docker-image predictions-combiner:latest --name mycluster
kind load docker-image web-app:latest --name mycluster

# Create ConfigMap for environmental variables
kubectl create configmap shared-config --from-env-file=.env

# Install and wait for the Ingress Nginx controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.2/deploy/static/provider/kind/deploy.yaml

echo "Waiting for Ingress Nginx controller to be ready..."
while ! kubectl get pods -n ingress-nginx --selector=app.kubernetes.io/component=controller | grep -q "1/1"; do
    sleep 10
done

# Apply the Kubernetes deployment configuration files dynamically
for f in deployment_configs/*-deployment-config.yaml; do
  kubectl apply -f "$f"
done

# Apply Ingress configuration
kubectl apply -f ingress-config-processed.yaml

# Display the pods and services
kubectl get pods
kubectl get services
kubectl get ingress