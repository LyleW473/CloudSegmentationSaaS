import numpy as np

from typing import Dict, Union
from src.backend.utils.exceptions import InvalidQueryException, PredictionsCombiningException

class CombinerPipeline:
    """
    The pipeline for combining the predicted segmentation masks for each of the extracted patches
    into a single segmentation mask for the entire image.
    """ 
    def _test_params(self, params:Dict[str, Union[np.ndarray, int]]) -> bool:
        """
        Tests whether the input parameters are valid for model inference.
        - The 'params' should be a standard Python dictionary.
        - It should contain the following keys:
            - "data": A numpy array of shape (num_patches, height, width)
            - "stride": The stride used to extract the patches.
            - "patch_size": The size of the patches extracted.
            - "original_img_height": The height of the original image.
            - "original_img_width": The width of the original image.
        """
        if not isinstance(params["data"], np.ndarray):
            try:
                params["data"] = np.array(params["data"])
            except:
                raise InvalidQueryException("The 'data' key should contain a numpy array or a Python list.")
        if not params["data"].ndim == 3:
            raise InvalidQueryException("The 'data' key should contain a 3 dimensions: (num_patches, height, width).")
        
    def __call__(self, params) -> Dict[str, np.ndarray]:
        """
        Given a dictionary with the relevant data, combines the predicted segmentation masks 
        for each of the extracted patches into a single segmentation mask for the entire image.

        Args:
            params (Dict[str, Union[np.ndarray, int]]): A dictionary mapping the data to be passed to the model.
                                                        e.g. {
                                                        "data": np.ndarray, 
                                                        "stride": int, 
                                                        "patch_size": int,
                                                        "original_img_height": int,
                                                        "original_img_width": int
                                                        }
        """
        # Validate the input parameters
        self._test_params(params)
        
        patch_size = params["patch_size"]
        stride = params["stride"]
        original_img_height = params["original_img_height"]
        original_img_width = params["original_img_width"]
        data = params["data"]

        # Combine the patches into a single segmentation mask
        try:
            canvas = np.zeros((original_img_height, original_img_width))
            index = 0
            for y in range(0, original_img_height, stride):
                for x in range(0, original_img_width, stride):
                    predicted_patch_mask = data[index]
                    index += 1

                    # Crop the patch if it extends beyond the image dimensions
                    mask_width = patch_size
                    mask_height = patch_size
                    if (x + patch_size > original_img_width):
                        mask_width = original_img_width - x
                    if (y + patch_size > original_img_height):
                        mask_height = original_img_height - y
                    predicted_patch_mask = predicted_patch_mask[:mask_height, :mask_width]

                    # Update the canvas with the patch mask (Choose the maximum value for each pixel)
                    canvas[y:y+mask_height, x:x+mask_width] = np.maximum(canvas[y:y+mask_height, x:x+mask_width], predicted_patch_mask)
            # Return combined mask as a list for JSON serialisation
            result = {"data": canvas.tolist(), "message": "Successfully combined the predicted segmentation masks."}
            return result
        except Exception as e:
            raise PredictionsCombiningException(f"An error occurred during predictions combining. {e}")