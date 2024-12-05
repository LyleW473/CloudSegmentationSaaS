# **Installation requirements for development** 
By installing the requirements, you should be able to test the entire application as well as train your own models. Please note that inside of the
setup scripts, specifying a GPU version of PyTorch may not be compatible with your system, which is why the CPU versions are installed instead. To train the 
models using CUDA, please modify the scripts to download a PyTorch version compatible with your system.

### Linux version
'./setup.sh'
### Windows version
'./setup.bat'

# **System requirements** 
Kind and Docker are expected to already be installed on the system. The requirements installed from the setup also require Python 3.11.9. **The installation
process will not work if not using Python 3.11.9.**

# **Setting up the '.env' file**
The variables used in the '.env' file are as follows:
- **MONGODB_URI** = The link to your MongoDB cluster, used for accessing the database.
- **TOKEN_GENERATOR_SECRET_KEY** = Any string that will be used as the secret key for token-based authentication
- **IS_TESTING** = A boolean indicating whether to use the production or test database. Set to 'false' for production database.
- **DATA_RETRIEVAL_HOST** = The host URL used for the data retrieval service, e.g, 'http://data-retrieval-service'
- **DATA_PROCESSING_HOST** = The host URL used for the data processing service, e.g, 'http://data-processing-service'
- **MODEL_INFERENCE_HOST** = The host URL used for the model inference service, e.g, 'http://model-inference-service'
- **PREDICTIONS_COMBINER_HOST** = The host URL used for the predictions combiner service, e.g, 'http://predictions-combiner-service'
- **WEB_APP_HOST** = The host URL used for the web-app service, e.g, 'http://web-app-service'
- **DATA_RETRIEVAL_PORT** = The port used for the data retrieval service, e.g., '8001'
- **DATA_PROCESSING_PORT** = The port used for the data processing service, e.g., '8002'
- **MODEL_INFERENCE_PORT** = The port used for the model inference service, e.g., '8003'
- **PREDICTIONS_COMBINER_PORT** = The port used for the predictions combiner service, e.g., '8004'
- **WEB_APP_PORT** = The port used for the web-app service, e.g., '8005'

If using GitHub Actions, you should set the GitHub secrets to be the exact same as your '.env' file.

# **1. Running the apps locally (without Docker)**
A script has been provided for running the apps locally which can be accessed via the links, dependent on what you set the 'WEB_APP_PORT' environmental variable to.  
'http://localhost:{WEB_APP_PORT}'  
or:  
'http://127.0.0.1:{WEB_APP_PORT}'  

### To run the script, just enter the command in your terminal:
'python scripts/run_apps.py'

# **2. Running the apps inside docker containers**
Assuming that you have installed the system requirements properly, a docker compose file has been provided for building all of the containers and running them.
Similar to **option 1**, you can access the system via the localhost at the specified web app port.
To build and run the containers, use the following commands:  
'docker-compose up --build'  
or:  
'docker compose up --build'  

# **3. Creating and running a Kubernetes cluster**
Instead of running the apps locally without or without docker, you also have the option of creating and running a Kubernetes Cluster with Ingress. 
If the containers do not already exist, the provided deployment script will build them for you. The system is accessible via the links:
'http://localhost'  
or:  
'http://localhost:80'    
or:  
'http://127.0.0.1:80' 
Use the following command to run the deployment script:

### Linux version
'chmod +x deploy.sh'
'./deploy.sh'

### Windows version
'./deploy.bat'

# **Instructions for querying the system**
Once you have accessed the web-page, you should be redirected to the login page.  
**1.** Enter any username and a password that has: At least 12 characters, contains a lowercase character, uppercase character, digit and a special character.  
**2.** Enter a site latitude in the range [-90, 90], site longitude in the range[-180, 180] and a time of interest in the format 'YYYY-MM-DD/YYYY-MM-DD'.  
**3.** Wait for the results to be generated.  
**4.** Interact with the navigation bar to see the generated data at different timesteps.  

# **Example queries**
**1. Statue of Liberty:** Site latitude = 40.6892, Site longitude = -74.0445, Time of interest = '2021-01-01/2024-01-02'  
**2. River Thames:** Site latitude = 51.5072, Site longitude = 0.1276, Time of interest = '2021-01-01/2024-01-02'  

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/pa_hoUiU)
