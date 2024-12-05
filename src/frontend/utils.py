import numpy as np
import os
import shutil

from PIL import Image
from typing import Dict, Any
from src.backend.settings.paths import MAIN_SAVED_IMAGES_DIR

def format_results():
    """
    Formats the results of the model inference pipeline to be displayed on the frontend.
    - Returns a list of dictionaries containing the visualisation and predicted mask image URLs,
      and more information for each date at the site location. 
    """
    results = []
    image_dir_names = os.listdir(f"{MAIN_SAVED_IMAGES_DIR}/images")
    image_dir_names.sort(key=lambda x: float(x.split(".png")[0].split("__")[-1])) # Sort by the cloud cover

    for file_name in image_dir_names:
        date, cloud_cover = file_name.split(".png")[0].split("__")
        visualisation_image_url = f"{MAIN_SAVED_IMAGES_DIR}/images/{file_name}"
        predicted_mask_image_url = f"{MAIN_SAVED_IMAGES_DIR}/masks/{file_name}"
        entry = {
            "visualisation_image_url": visualisation_image_url, 
            "predicted_mask_image_url": predicted_mask_image_url,
            "date": date,
            "cloud_coverage": cloud_cover
            }
        results.append(entry)
    return results

def save_results(result:Dict[str, Any]) -> None:
    """
    Saves the visusalisation image and the predicted segmentation mask to disk.
    - Will be saved in the "saved_images" directory.
    - Will be accessible via the "result" endpoint, displaying the images on the
      frontend.

    Args:
        result (Dict[str, Any]): A Python dictionary containing the results of the
                                model inference pipeline. Should contain the keys:
                                - "predicted_mask": The predicted segmentation mask.
                                - "visualisation_image": The visualisation image.
                                - "raw_processed_image": The raw processed image.
                                - "raw_patches": The raw patches used for inference.
    """
    # Check if the required keys are present
    successful_retrieval_keys = [
                                "predicted_masks", 
                                "visualisation_images", 
                                "raw_processed_images", 
                                "raw_patches",
                                "metadatas"
                                ]
    flag = True
    for key in successful_retrieval_keys:
        if key not in result:
            flag = False
            break
    if not flag:
        raise ValueError(f"Missing keys in the result dictionary: {[key for key in successful_retrieval_keys if key not in result]}")
    
    # Save the images for each timestep
    num_timesteps = len(result["predicted_masks"])
    predicted_masks = result["predicted_masks"]
    visualisation_images = result["visualisation_images"]
    # raw_processed_images = result["raw_processed_images"]
    # all_raw_patches = result["raw_patches"]
    metadatas = result["metadatas"]

    # Clear the saved images directory (To remove any previous results stored)
    shutil.rmtree(MAIN_SAVED_IMAGES_DIR)
    os.makedirs(MAIN_SAVED_IMAGES_DIR)
    os.makedirs(f"{MAIN_SAVED_IMAGES_DIR}/images")
    os.makedirs(f"{MAIN_SAVED_IMAGES_DIR}/masks")

    for time_step in range(0, num_timesteps):
        # Convert the results to numpy arrays
        predicted_mask = np.array(predicted_masks[time_step])
        visualisation_image = np.array(visualisation_images[time_step])
        # raw_processed_image = np.array(raw_processed_images[time_step])
        # raw_patches = np.array(all_raw_patches[time_step])

        # Extract metadata
        cloud_cover = metadatas[time_step]["cloud_coverage"]
        date = metadatas[time_step]["date"]

        name = f"{date}__{cloud_cover}"

        # Save the visualisation image and the predicted segmentation mask
        visualisation_image = Image.fromarray(visualisation_image.astype(np.uint8))
        visualisation_image.save(f"{MAIN_SAVED_IMAGES_DIR}/images/{name}.png")

        predicted_mask_image = Image.fromarray((predicted_mask * 255).astype(np.uint8))
        predicted_mask_image.save(f"{MAIN_SAVED_IMAGES_DIR}/masks/{name}.png")