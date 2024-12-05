
FROM python:3.11-slim

WORKDIR /project

# Copy requirements.txt and setup script from the root project directory
COPY apps/predictions_combiner/requirements.txt /project/apps/predictions_combiner/

# Copy the rest of the application code
COPY src/ /project/src/
COPY apps/predictions_combiner /project/apps/predictions_combiner

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r apps/predictions_combiner/requirements.txt

# List installed Python packages to verify setup
RUN pip list

# Set the PYTHONPATH environment variable
# Add the project directory and the src and apps directories to the Python path
ENV PYTHONPATH="/project:/project/src:/project/apps"

# Run the application
CMD ["sh", "-c", "ls && pip list && python apps/predictions_combiner/app_predictions_combiner.py"]