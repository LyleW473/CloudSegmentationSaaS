@echo off

REM Install python-dotenv (For running the scripts)
pip install python-dotenv --no-cache-dir

REM Run the scripts to create the configuration files
python scripts\create_cluster_config.py
python scripts\create_ingress_config.py
python scripts\create_deployment_config.py

REM Set up the kind cluster
kind create cluster --name mycluster --config cluster-config-processed.yaml --wait 5m
kubectl cluster-info --context kind-mycluster
kubectl get nodes

REM Build the Docker images
docker compose build

REM Load individual Docker images into the kind cluster
kind load docker-image data-retrieval:latest --name mycluster
kind load docker-image data-processing:latest --name mycluster
kind load docker-image model-inference:latest --name mycluster
kind load docker-image predictions-combiner:latest --name mycluster
kind load docker-image web-app:latest --name mycluster

REM Create ConfigMap for environmental variables
kubectl create configmap shared-config --from-env-file=.env

REM Install and wait for the Ingress Nginx controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.2/deploy/static/provider/kind/deploy.yaml

echo Waiting for Ingress Nginx controller to be ready...
:wait_loop
kubectl get pods -n ingress-nginx --selector=app.kubernetes.io/component=controller | findstr /i "1/1" >nul
if %errorlevel% neq 0 (
    timeout /t 10 >nul
    goto wait_loop
)

REM Apply the Kubernetes deployment configuration files dynamically
for %%f in (deployment_configs\*-deployment-config.yaml) do (
  kubectl apply -f "%%f"
)

REM Apply Ingress configuration
kubectl apply -f ingress-config-processed.yaml

REM Display the pods and services
kubectl get pods
kubectl get services
kubectl get ingress