"""
Creates the deployment configuration files for each container in the system.
It will substitute the placeholders inside the deployment configuration template 
with the actual values set right before writing the processed content to a new file.

- Automates the process of creating the deployment configuration files for each container.
- Prevents the need to manually update the deployment configuration template with the actual values.
- Prevents hardcoding the values in the file (and exposing the container ports to the public).
"""
import set_path
import os
from string import Template
from dotenv import load_dotenv
load_dotenv()

containers = {
            "data-retrieval": {"port": os.getenv("DATA_RETRIEVAL_PORT"), "num_replicas": 2},
            "data-processing": {"port": os.getenv("DATA_PROCESSING_PORT"), "num_replicas": 2},
            "model-inference": {"port": os.getenv("MODEL_INFERENCE_PORT"), "num_replicas": 4},
            "predictions-combiner": {"port": os.getenv("PREDICTIONS_COMBINER_PORT"), "num_replicas": 2},
            "web-app": {"port": os.getenv("WEB_APP_PORT"), "num_replicas": 1} # Only a single replica, NFS not supported so errors will occur
            }

for container_name, info_dict in containers.items():
    port = str(info_dict["port"])
    print(f"Container: {container_name}, Port: {port}")

    # Read the deployment configuration template
    with open('deployment-config-template.yaml', 'r') as file:
        template = Template(file.read())
    
    replaced_container_name = container_name.replace("_", "-") # e.g., "web_app" -> "web-app"
    num_replicas = info_dict["num_replicas"]
    
    # Set the environment variables that will be substituted in the template
    os.environ["CONTAINER_PORT"] = port
    os.environ["SERVICE_PORT"] = port # Expose on same port as container
    os.environ["APP_NAME"] = replaced_container_name 
    os.environ["CONTAINER_IMAGE"] = f"{container_name}:latest" # e.g., "orchestration:latest"
    os.environ["NUM_REPLICAS"] = str(num_replicas)

    # Substitute the environment variables in the template
    config = template.substitute(os.environ)

    os.makedirs("deployment_configs", exist_ok=True)
    with open(f"deployment_configs/{replaced_container_name}-deployment-config.yaml", 'w') as file:
        file.write(config)