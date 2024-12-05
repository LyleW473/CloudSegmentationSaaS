import os
from dotenv import load_dotenv

load_dotenv()
MAIN_SAVED_IMAGES_DIR = "saved_images"
TEMPLATES_DIR = "src/frontend/templates"

ENDPOINT_URLS = {
    "data_retrieval": {
        "base_url": f"{os.getenv('DATA_RETRIEVAL_HOST')}:{os.getenv('DATA_RETRIEVAL_PORT')}",
        "app_name": "app_data_retrieval",
        "path": "/data_retrieval"
    },
    "data_processing": {
        "base_url": f"{os.getenv('DATA_PROCESSING_HOST')}:{os.getenv('DATA_PROCESSING_PORT')}",
        "app_name": "app_data_processing",
        "path": "/data_processing"
    },
    "model_inference": {
        "base_url": f"{os.getenv('MODEL_INFERENCE_HOST')}:{os.getenv('MODEL_INFERENCE_PORT')}",
        "app_name": "app_model_inference",
        "path": "/model_inference"
    },
    "predictions_combiner": {
        "base_url": f"{os.getenv('PREDICTIONS_COMBINER_HOST')}:{os.getenv('PREDICTIONS_COMBINER_PORT')}",
        "app_name": "app_predictions_combiner",
        "path": "/predictions_combiner"
    },
    "web_app": {
        "base_url": f"{os.getenv('WEB_APP_HOST')}:{os.getenv('WEB_APP_PORT')}",
        "app_name": "app_frontend",
        "path": "/", # For convention
        "additional_paths": {
                            "login": "/login",
                            "user_authentication": "/user_authentication",
                            "query": "/query",
                            "result": "/result"
        }
    }
}