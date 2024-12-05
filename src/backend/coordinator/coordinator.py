import numpy as np
import requests
import time
import logging

from typing import Any, Dict, Tuple
from fastapi import status

class PipelineCoordinator:
    """
    This class is used to coordinate the retrieval and processing pipelines
    for the Landsat-8 satellite imagery data. It will ensure that the input
    parameters are valid before passing them to the respective pipelines
    (with the pipeline-specific checks being performed within each pipeline).

    The class will return the results in a formatted response.
    """

    def __init__(self, pipeline_urls:Dict[str, str]):
        self.pipeline_urls = pipeline_urls
        self.log = logging.getLogger(self.__class__.__name__)
    
    def build_request(self, url:str, data:Dict[str, Any], headers:Dict[str, str]) -> Dict[str, Any]:
        """
        Builds the request dictionary for the requests.post method. (Helper function)

        Args:
            url (str): The URL to which the request is to be made.
            data (Dict[str, Any]): The data to be sent in the request.
            headers (Dict[str, str]): The headers to be sent in the request.
        """
        return {"url": url, "json": data, "headers": headers}
        
    def execute(self, parameters_dict:Dict[str, Any], headers:Dict[str, str]) -> Tuple[Dict[str, Any], int]:
        """
        Given a dictionary of input parameters, coordinates the retrieval, processing and
        inference pipelines for the Landsat-8 satellite imagery data.
        
        Args:
            parameters_dict (Dict[str, Any]): The dictionary of input parameters.
                                            The dictionary should contain the keys:
                                            - "site_latitude": The latitude of the site of interest.
                                            - "site_longitude": The longitude of the site of interest.
                                            - "time_of_interest": The time range of interest in the format 
                                                                  "YYYY-MM-DD/YYYY-MM-DD".
            headers (Dict[str, str]): The headers to be sent with each request. Should contain
                                      an authorization token to be able to access the endpoints.
        """
        start_time = time.perf_counter()

        # Data retrieval
        url = self.pipeline_urls["data_retrieval"]
        self.log.info(f"Performing data retrieval [1/4] ...")
        data = {**parameters_dict}
        result1 = requests.post(**self.build_request(url, data, headers))
        if result1.status_code != status.HTTP_200_OK:
            return result1.json(), result1.status_code
        result1 = result1.json()
        self.log.info("Data retrieval successful [1/4] ...")

        # Process each time-step separately, due to limitations on the amount of data that can be processed at once
        extracted_bands_dicts_list = result1["data"]
        all_metadatas = result1["metadatas"]
        all_patches = []
        all_predicted_masks = []
        all_visualisation_images = []
        all_processed_images = []

        num_items = len(extracted_bands_dicts_list)
        for i, extracted_bands_dict in enumerate(extracted_bands_dicts_list):

            item_num = i + 1

            # Data processing
            self.log.info(f"Performing data processing [2/4] for item [{item_num}/{num_items}] ...")
            url = self.pipeline_urls["data_processing"]
            data = {"data": extracted_bands_dict}
            result2 = requests.post(**self.build_request(url, data, headers))
            if result2.status_code != status.HTTP_200_OK:
                return result2.json(), result2.status_code
            result2 = result2.json()
            self.log.info(f"Data processing successful [2/4] for item [{item_num}/{num_items}] ...")

            # Model inference
            self.log.info(f"Performing model inference [3/4] for item [{item_num}/{num_items}] ...")
            url = self.pipeline_urls["model_inference"]
            data = {"data": result2["data"]["patches"]}
            result3 = requests.post(**self.build_request(url, data, headers))
            if result3.status_code != status.HTTP_200_OK:
                return result3.json(), result3.status_code
            result3 = result3.json()
            self.log.info(f"Model inference successful [3/4] for item [{item_num}/{num_items}] ...")
            
            # Combining predictions (all patches to a single mask)
            self.log.info(f"Combining predictions [4/4] for item [{item_num}/{num_items}] ...")
            N, H, W = np.array(result3["masks"]).astype(np.float32).shape
            original_img_height, original_img_width = np.array(extracted_bands_dict["red"]).shape
            combiner_args = {
                            "data": result3["masks"],
                            "stride": H, 
                            "patch_size": H, 
                            "original_img_height": original_img_height, 
                            "original_img_width": original_img_width
                            }
            url = self.pipeline_urls["predictions_combiner"]
            result4 = requests.post(**self.build_request(url, combiner_args, headers))
            if result4.status_code != status.HTTP_200_OK:
                return result4.json(), result4.status_code
            result4 = result4.json()
            self.log.info(f"Combining predictions successful [4/4] for item [{item_num}/{num_items}] ...")

            # Append the results
            all_patches.append(result2["data"]["patches"])
            all_predicted_masks.append(result4["data"])
            all_visualisation_images.append(result2["data"]["visualisation_image"])
            all_processed_images.append(result2["data"]["processed_image"])

        self.log.info("Pipeline run successful, returning the results ...")
        end_time = time.perf_counter()
        data = {   
                "time_taken": end_time - start_time,
                "predicted_masks": all_predicted_masks,
                "visualisation_images": all_visualisation_images,
                "raw_processed_images": all_processed_images,
                "raw_patches": all_patches,
                "metadatas": all_metadatas,
                "message": "Successful pipeline run.",
                }
        return data, status.HTTP_200_OK