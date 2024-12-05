
FROM python:3.11-slim

WORKDIR /project

# Copy requirements.txt and setup script from the root project directory
COPY apps/data_processing/requirements.txt /project/apps/data_processing/

# Copy the rest of the application code
COPY src/ /project/src/
COPY apps/data_processing /project/apps/data_processing

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r apps/data_processing/requirements.txt

# List installed Python packages to verify setup
RUN pip list

# Set the PYTHONPATH environment variable
# Add the project directory and the src and apps directories to the Python path
ENV PYTHONPATH="/project:/project/src:/project/apps"

# Run the application
CMD ["sh", "-c", "ls && pip list && python apps/data_processing/app_data_processing.py"]