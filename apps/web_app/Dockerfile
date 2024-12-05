
FROM python:3.11-slim

WORKDIR /project

# Copy requirements.txt and setup script from the root project directory
COPY apps/web_app/requirements.txt /project/apps/web_app/

# Copy the rest of the application code
COPY src/ /project/src/
COPY apps/web_app /project/apps/web_app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r apps/web_app/requirements.txt

# List installed Python packages to verify setup
RUN pip list

# Set the PYTHONPATH environment variable
# Add the project directory and the src and apps directories to the Python path
ENV PYTHONPATH="/project:/project/src:/project/apps"

# Run the application
CMD ["sh", "-c", "ls && pip list && python apps/web_app/app_frontend.py"]