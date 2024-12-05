
FROM python:3.11-slim

WORKDIR /project

# Install system dependencies
# ImportError: libexpat.so.1: cannot open shared object file: No such file or directory
RUN apt-get update && apt-get install -y \
    libexpat1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and setup script from the root project directory
COPY apps/data_retrieval/requirements.txt /project/apps/data_retrieval/

# Copy the rest of the application code
COPY src/ /project/src/
COPY apps/data_retrieval /project/apps/data_retrieval

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r apps/data_retrieval/requirements.txt

# List installed Python packages to verify setup
RUN pip list

# Set the PYTHONPATH environment variable
# Add the project directory and the src and apps directories to the Python path
ENV PYTHONPATH="/project:/project/src:/project/apps"

# Run the application
CMD ["sh", "-c", "ls && pip list && python apps/data_retrieval/app_data_retrieval.py"]