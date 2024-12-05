import os
import pandas as pd
import numpy as np
from PIL import Image

class SatelliteImageProcessor:
    def __init__(self):
        """
        Initialises the SatelliteImageProcessor class, which is used
        to create normalised (R, G, B, NIR) images.
        """
        self.interested_bands = ["red", "green", "blue", "nir"]

    def create_rgbnir_images(self, base_dir:str, patch_csv_file_name:str, is_train:bool) -> None:
        """
        Creates (R, G, B, NIR) images for all patch images inside of a specified
        dataset. The images are saved in a new directory called "train_rgbnir" within
        that dataset directory.

        Args:
            base_dir (str): The base directory of the dataset.
            patch_csv_file_name (str): The name of the CSV file containing the patch names.
            is_train (bool): Whether the images are training images.
        """

        patch_names = pd.read_csv(f"{base_dir}/{patch_csv_file_name}")["name"].tolist()
        print(patch_names)

        prefix = "train" if is_train else "test"
        new_images_dir = f"{base_dir}/{prefix}_rgbnir"
        if os.path.exists(new_images_dir):
            print(f"Directory {new_images_dir} already exists. Skipping the creation of new images.")
            return
        os.makedirs(new_images_dir, exist_ok=True)
        
        print(f"Creating new images in {new_images_dir}...")
        for patch_name in patch_names:
            band_arrays = self._create_rgbnir_image(
                                                    base_dir=base_dir, 
                                                    patch_name=patch_name,
                                                    is_train=is_train
                                                    )
            # Normalize from [0, (2^16)-1] to [0, 255]
            band_arrays = (band_arrays / (2**16 - 1)) * 255

            # Convert to uint8
            band_arrays = band_arrays.astype(np.uint8)

            # Save the new image to the new directory
            new_image_path = f"{new_images_dir}/rgbnir_{patch_name}.tif"
            new_image = Image.fromarray(band_arrays)
            new_image.save(new_image_path)

    def _create_rgbnir_image(self, base_dir:str, patch_name:str, is_train:bool) -> np.ndarray:
        """
        Creates a single 4-channel image (R, G, B, NIR) by combining
        the 4 bands into a single image.

        Args:
            base_dir (str): The base directory of the dataset.
            patch_name (str): The name of the patch.
            is_train (bool): Whether the image is a training image.
        """
        prefix = "train" if is_train else "test"
        band_arrays = []
        for band in self.interested_bands:
            # e.g., data/archive/38-Cloud_training/train_red/red_00000000.TIF
            image_path = f"{base_dir}/{prefix}_{band}/{band}_{patch_name}.TIF"
            band_data = Image.open(image_path)
            band_data = np.array(band_data)
            band_arrays.append(band_data)
        band_arrays = np.stack(band_arrays, axis=-1) # (C, H, W)
        return band_arrays