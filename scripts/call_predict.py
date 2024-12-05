import set_path
import requests
import numpy as np
import os
import time

from typing import Dict, Any
from PIL import Image

def call_predict(latitude:float, longitude:float, time_of_interest:str) -> Dict[str, Any]:
    """
    Function for calling the FastAPI endpoint for model inference.

    Args:
        latitude (float): The latitude of the site location (assumed to be the center of the image).
        longitude (float): The longitude of the site location (assumed to be the center of the image).
        time_of_interest (str): The time period of interest in the format "YYYY-MM-DD/YYYY-MM-DD".
    """
    url = "http://127.0.0.1:8000/predict" # TODO: Change this to the VM's IP address
    query = {"site_latitude": latitude, "site_longitude": longitude, "time_of_interest": time_of_interest}
    response = requests.post(url, json=query)
    return response.json()

if __name__ == "__main__":
    # Queries
    test_locations = {
                    "Statue of Liberty": (40.6892, -74.0445), 
                    "River Thames": (51.5072, 0.1276),
                    "Invalid location": (200, -74.0445)
                    }
    time_of_interest = "2021-01-01/2024-01-02"

    successful_retrieval_keys = ["predicted_mask", "visualisation_image", "raw_processed_image", "raw_patches"]
    for location, (latitude, longitude) in test_locations.items():
        start_time = time.perf_counter()
        print(f"Querying for {location} at latitude {latitude}, longitude {longitude}")

        result = call_predict(latitude, longitude, time_of_interest)
        # print(result.keys())
        # patches = np.array(result["data"]["patches"])
        # visualisation_image = np.array(result["data"]["visualisation_image"])
        # print("V", visualisation_image.shape)
        # print("P", patches.shape)
        # print(result["message"])
        print(f"Time taken for location {location}: {time.perf_counter() - start_time:.4f} seconds.")

        flag = True
        for key in successful_retrieval_keys:
            if key not in result:
                flag = False
                break
        if not flag:
            print(f"Failed to retrieve data for {location}")
            continue

        predicted_mask = np.array(result["predicted_mask"])
        visualisation_image = np.array(result["visualisation_image"])
        raw_processed_image = np.array(result["raw_processed_image"])
        raw_patches = np.array(result["raw_patches"])
        print("Predicted mask", predicted_mask.shape)
        print("Visualisation image", visualisation_image.shape)
        print("Raw processed image", raw_processed_image.shape)
        print("Raw patches", raw_patches.shape)

        # Save results 
        os.makedirs("saved_images", exist_ok=True)
        print(visualisation_image.flatten().min(), visualisation_image.flatten().max())
        visualisation_image = Image.fromarray(visualisation_image.astype(np.uint8))
        visualisation_image.save(f"saved_images/{location}_image.png")

        print(predicted_mask.flatten().min(), predicted_mask.flatten().max())
        predicted_mask_image = Image.fromarray((predicted_mask * 255).astype(np.uint8))
        predicted_mask_image.save(f"saved_images/{location}_mask.png")