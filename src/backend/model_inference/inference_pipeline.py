import numpy as np

from typing import Dict, Union, List
from src.backend.model_inference.utils import get_model_predictions
from src.backend.utils.exceptions import InvalidQueryException

class InferencePipeline:
    """
    The core pipeline for passing the pre-processed data to the model for inference.
    """ 
    def _test_params(self, params:Dict[str, np.ndarray]) -> bool:
        """
        Tests whether the input parameters are valid for model inference.
        - The 'params' should be a standard Python dictionary.
        - It should contain the key: "data" which is a numpy array of shape (num_patches, height, width, num_channels).
        """
        if not isinstance(params, dict):
            raise InvalidQueryException("The input parameters should be a Python dictionary.")
        if "data" not in params:
            raise InvalidQueryException("The 'data' key should be present in the input parameters.")
        if not isinstance(params["data"], np.ndarray):
            try:
                params["data"] = np.array(params["data"])
            except:
                raise InvalidQueryException("The 'data' key should contain a numpy array or a Python list.")
        if not params["data"].ndim == 4:
            raise InvalidQueryException("The 'data' key should contain have 4 dimensions: (num_patches, height, width, num_channels).")
        
    def __call__(self, params:Dict[str, Union[List, np.ndarray]]) -> Dict[str, Union[None, Dict[str, np.ndarray]]]:
        """
        Given a dictionary mapping {"data": np.ndarray}, passes the data to the 
        model for inference, returning the generated segmentation masks from 
        the model.

        Args:
            params (Dict[str, Union[List, np.ndarray]]): A dictionary mapping the data to be passed to the model.
                                            e.g. {"data": np.ndarray}
        """
        # Validate the input parameters
        self._test_params(params)
     
        data = params["data"]
        predictions = get_model_predictions(data)

        # Return the results
        result = {"data": data.tolist()}

        # Note: Convert the predictions to a list for JSON serialisation
        if predictions is None:
            result.update({"masks": None, "message": "Model inference failed."})
        else:
            result.update({"masks": predictions.tolist(), "message": "Successful model inference."})
        return result