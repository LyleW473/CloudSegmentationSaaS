import numpy as np

from src.backend.data_processing import *
from src.backend.utils.exceptions import InvalidQueryException
from typing import Dict, Tuple, Union, List

class ProcessingPipeline:
    """
    The core pipeline for processing the retrieved Landsat-8 satellite imagery:
    - Combines the extracted bands into a single 4-channel image.
    - Converts the combined image into patches of a specified size.

    The pipeline returns a dictionary mapping: {"data": np.ndarray} where the 
    np.ndarray is the patches of the combined image.
    """ 
    def __init__(self, patch_size:Tuple[int, int]=(384, 384)):
        """
        Initialises the ProcessingPipeline with the desired patch size.
        
        Args:
            patch_size (Tuple[int, int]): The size of the patches to be extracted from the combined image.
        """
        self.band_combiner = BandCombiner()
        self.patchify = Patchify()
        self.desired_patch_size = patch_size

    def test_params(self, query:Dict[str, np.ndarray]) -> bool:
        """
        Tests whether the query contains the required keys and values for processing.
        - The 'query' should be a Python dictionary containing the key "data", containing the extracted bands.
        - The 'bands_dict' should be a standard Python dictionary.
        - The 'bands_dict' should contain the keys: "red", "green", "blue", "nir08".
        - The values should be numpy arrays of shape (height, width)
        """
        bands_dict = query.get("data", None)
        
        for key in ["red", "green", "blue", "nir08"]:
            if key not in bands_dict:
                raise InvalidQueryException(f"The '{key}' key should be present in the 'data' dictionary.")
        
        for key, val in bands_dict.items():
            if not isinstance(val, np.ndarray):
                try:
                    bands_dict[key] = np.array(val)
                except:
                    raise InvalidQueryException(f"The '{key}' key should contain a numpy array.")
            if bands_dict[key].ndim != 2:
                raise InvalidQueryException(f"The '{key}' key should contain a 2D numpy array.")
        return True
    
    def __call__(self, query:Dict[str, Dict[str, Union[List, np.ndarray]]]) -> Dict[str, np.ndarray]:
        """
        Given a dictionary mapping the bands to their respective numpy arrays, processes the data.
        - Validates the input parameters.
        - Combines the bands into a single image.
        - Converts the combined image into patches of the desired size.

        Args:
            query (Dict[str, Dict[str, Union[List, np.ndarray]]]): A dictionary mapping "data" to a dictionary of the extracted bands.
                                            The extracted bands should be numpy arrays of shape (height, width) or Python lists of
                                            numbers.
        """

        # Test the input parameters
        self.test_params(query)
        bands_dict = query["data"]

        # Combine all of the bands into a single image
        combined_image = self.band_combiner(extracted_bands_dict=bands_dict)

        # Normalise data to [0, 1]
        combined_image = combined_image / (2**16 - 1)

        # Image for visualisation
        visualisation_image = (combined_image.copy() * 255).astype(np.uint8)[:, :, :3] # (R, G, B) only

        # Convert the combined image to patches, padding the image if necessary
        patches = self.patchify(combined_image, patch_size=self.desired_patch_size)

        # Convert to list for JSON serialisation
        visualisation_image = visualisation_image.tolist()
        combined_image = combined_image.tolist()
        patches = patches.tolist()

        data = {
                "visualisation_image": visualisation_image,
                "processed_image": combined_image,
                "patches": patches
                }
        return {"data": data, "message": "Successful data processing."} 