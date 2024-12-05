
FROM python:3.11-slim

WORKDIR /project

# Copy requirements.txt and setup script from the root project directory
COPY apps/model_inference/requirements.txt /project/apps/model_inference/

# Copy the model weights and dataset statistics
COPY model_weights/ /project/model_weights/
COPY dataset_statistics/ /project/dataset_statistics/

# Copy the rest of the application code
COPY src/ /project/src/
COPY apps/model_inference /project/apps/model_inference

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r apps/model_inference/requirements.txt

# List installed Python packages to verify setup
RUN pip list

# Set the PYTHONPATH environment variable
# Add the project directory and the src and apps directories to the Python path
ENV PYTHONPATH="/project:/project/src:/project/apps"

# Run the application
CMD ["sh", "-c", "ls && pip list && python apps/model_inference/app_model_inference.py"]