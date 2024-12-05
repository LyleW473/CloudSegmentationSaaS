import numpy as np

from typing import Dict

class BandCombiner:
    def __call__(self, extracted_bands_dict:Dict[str, np.ndarray]) -> np.ndarray:
        """
        Combines the extracted bands into a single 4-channel image.
        - Returns in the order: (R, G, B, NIR)
        - Returns the image in the format (height, width, channels)

        Args:
            extracted_bands_dict (Dict[str, np.ndarray]): A dictionary mapping the band names to numpy arrays of the band data.
        """
        red_band = extracted_bands_dict["red"]
        green_band = extracted_bands_dict["green"]
        blue_band = extracted_bands_dict["blue"]
        nir_band = extracted_bands_dict["nir08"]
        return np.stack([red_band, green_band, blue_band, nir_band], axis=-1)