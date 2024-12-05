"""
This script reads the Ingress template file, replaces placeholders with environment variables, and writes the processed config to a new file.
"""
import set_path
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Read the Ingress template
with open("ingress-config-template.yaml", "r") as file:
    content = file.read()

# Replace placeholders with environment variables
processed_content = content
for key, value in os.environ.items():
    placeholder = f"${{{key}}}"
    processed_content = processed_content.replace(placeholder, value)

# Write the processed config to a new file
with open("ingress-config-processed.yaml", "w") as file:
    file.write(processed_content)

print("Processed Ingress config written to ingress-config-processed.yaml")