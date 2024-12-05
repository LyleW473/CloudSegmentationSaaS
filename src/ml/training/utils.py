import os
import numpy as np
import json

from typing import List, Union, Tuple
from PIL import Image

def get_image_paths(directory:str) -> List[str]:
    """
    Returns a list of file paths to images in a specified directory.

    Args:
        directory (str): The directory containing the images.
    """
    return [os.path.join(directory, f) for f in os.listdir(directory)]

def copy_weights_to_new_channels(weights:Union[List[float], np.ndarray], in_channels:int) -> np.ndarray:
    """
    Copies the weights of a model to new channels.
    - For example if the model was trained on 3 channels and now needs to be trained on 4 channels,
      the dataset mean and std used to normalise the data will need to be copied to the new channel.
    """
    repeated_weights = np.tile(weights, in_channels // len(weights) + 1)
    return repeated_weights[:in_channels]

def calculate_dataset_statistics(image_paths:List[str]) -> Tuple[List[float], List[float]]:
    """
    Calculates the mean and standard deviation of the dataset and
    saves it as a JSON file.

    Args:
        image_paths (List[str]): A list of image file paths used to calculate the statistics.
    """
    if os.path.exists("dataset_statistics/stats.json"):
        print("Dataset statistics already calculated.")
        with open("dataset_statistics/stats.json", "r") as f:
            stats = json.load(f)
        return stats["mean"], stats["std"]

    num_channels = 4
    channel_sums = np.zeros(num_channels)
    channel_squared_sums = np.zeros(num_channels)
    channel_maxs = np.ones(num_channels) * -np.inf
    channel_mins = np.ones(num_channels) * np.inf
    num_pixels = 0

    num_images = len(image_paths)
    for i, image_path in enumerate(image_paths):
        print(f"Processing image {i + 1}/{num_images} | Progress: {((i + 1) / num_images) * 100:.5f}%")
        image = Image.open(image_path)
        image = np.array(image)
        image = image / 255.0 # Normalise to [0, 1] first (The normalisation will be applied to [0, 1] images)
        image = np.transpose(image, (2, 0, 1)) # [H, W, C] -> [C, H, W]

        channel_sums += np.sum(image, axis=(1, 2))
        channel_squared_sums += np.sum(image ** 2, axis=(1, 2))
        num_pixels += image.shape[1] * image.shape[2] # H * W

        channel_maxs = np.maximum(channel_maxs, np.max(image, axis=(1, 2)))
        channel_mins = np.minimum(channel_mins, np.min(image, axis=(1, 2)))

    channel_means = channel_sums / num_pixels
    channel_std_devs = np.sqrt((channel_squared_sums / num_pixels) - channel_means ** 2)

    # Save the statistics
    os.makedirs("dataset_statistics", exist_ok=True)
    means = channel_means.tolist()
    std_devs = channel_std_devs.tolist()
    with open("dataset_statistics/stats.json", "w") as f:
        json.dump({"mean": means, "std": std_devs}, f)
    return means, std_devs