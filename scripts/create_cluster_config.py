"""
Substitutes the placeholders in the cluster-config-template.yaml file with the actual values from the environment variables
(defined in the .env file) and writes the processed content to a new file called cluster-config-processed.yaml.

- This is used to prevent the need to manually update the cluster-config-template.yaml file with the actual values
  and also to prevent hardcoding the values in the file (and exposing the container ports to the public).
"""
import set_path
import os
from dotenv import load_dotenv

load_dotenv()

# Read the cluster config template
with open("cluster-config-template.yaml", "r") as file:
    content = file.read()

# Replace placeholders with environment variables
processed_content = content
for key, value in os.environ.items():
    placeholder = f"${{{key}}}"
    processed_content = processed_content.replace(placeholder, value)

# Write the processed config to a new file
with open("cluster-config-processed.yaml", "w") as file:
    file.write(processed_content)

print("Processed cluster config written to cluster-config-processed.yaml")
