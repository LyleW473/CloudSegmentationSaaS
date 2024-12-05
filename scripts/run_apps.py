"""
Script for running all of the applications locally, without
containerising them. This is useful for testing the applications
locally, without needing to build and run the Docker containers.
"""
import set_path
import subprocess
import os
from dotenv import load_dotenv
from src.backend.settings.paths import ENDPOINT_URLS

# Is testing, e.g., CI/CD pipeline with GitHub Actions
load_dotenv()
os.environ["IS_TESTING"] = os.getenv("IS_TESTING", "false").lower()

# Construct the commands for running the FastAPI applications
commands = []
for service, info_dict in ENDPOINT_URLS.items():
    app_name = info_dict.get("app_name")
    base_url = info_dict.get("base_url")
    
    # Extract the root URL and port
    # E.g., "http://127.0.0.1:8003" -> ["127.0.0.1", "8003"]
    data = base_url.split("://")[-1]
    _, port = data.split(":")

    # Set internal environment variables for the services (e.g., the ENDPOINT_URLS dictionary needs to be updated with these local URLs)
    os.environ[f"{service.upper()}_HOST"] = "http://127.0.0.1" # e.g., DATA_RETRIEVAL_HOST=http://127.0.0.1
    command = f"uvicorn apps.{service}.{app_name}:app --host 127.0.0.1 --port {port}" # Always run locally
    commands.append(command)

if os.name == "nt": # Windows
    for command in commands:
        subprocess.Popen(["start", "cmd", "/k", command], shell=True)
else:
    # Linux or MacOS
    for command in commands:
        subprocess.Popen(command, shell=True)