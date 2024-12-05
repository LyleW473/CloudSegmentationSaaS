import numpy as np

from src.backend.data_retrieval.data_retriever import DataRetriever
from src.backend.utils.exceptions import InvalidQueryException, DataNotFoundException
from src.backend.data_processing import *
from typing import Dict, Union
from datetime import datetime

class RetrievalPipeline:
    """
    The core pipeline for retrieving Landsat-8 satellite imagery based on the
    query parameters provided. The pipeline returns either:
    - A dictionary mapping: {"data": {"red": np.ndarray, "green": np.ndarray, "blue": np.ndarray, "nir08": np.ndarray}}
    - A dictionary mapping: {"data": None} if no data is found.
    """ 
    def __init__(self):
        self.data_retriever = DataRetriever() 
    
    def _test_params(self, query:Dict[str, Union[float, str]]) -> None:
        """
        Tests whether the input parameters are valid for data retrieval.

        - Should contain the following keys:
            - site_latitude (float): The latitude of the site of interest.
            - site_longitude (float): The longitude of the site of interest.
            - time_of_interest (str): The time range of interest in the format "YYYY-MM-DD/YYYY-MM-DD".
        - The 'site_latitude' and 'site_longitude' should be floats or integers.
        - The 'site_latitude' should be in the range [-90, 90].
        - The 'site_longitude' should be in the range [-180, 180].
        - The 'time_of_interest' should be a string in the format "YYYY-MM-DD/YYYY-MM-DD".
        - The 'time_of_interest' should exist
        - The start date should be before the end date.

        - Raises an exception which is expected to be caught by the caller.
        - The basic input validation is performed at the endpoint that calls this pipeline.

        Args:
            query (Dict[str, Union[float, str]]): The dictionary of input parameters for data retrieval.
        """

        time_of_interest = query.get("time_of_interest")
        
        # Ensure that the start date is before the end date
        start_date_str, end_date_str = time_of_interest.split("/")
        try:
            # Error: Could be an error in the conversion, e.g., if it does not exist
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
        except ValueError:
            raise InvalidQueryException("Invalid date, try a different date, this date may not exist.")
        
        if start_date > end_date:
            raise InvalidQueryException("The start date should be before the end date.")
        
    def __call__(self, query:Dict[str, Union[float, str]]) -> Dict[str, Union[str, np.ndarray]]:
        """
        Given a site of interest (lat, lon), retrieves the least cloudy Landsat-8 image and extracts the bands of 
        interest from the image.

        Args:
            query: The dictionary of input parameters for data retrieval.
        """
        # Validate the input parameters
        self._test_params(query)

        # Data retrieval
        site_latitude = query["site_latitude"]
        site_longitude = query["site_longitude"]
        time_of_interest = query["time_of_interest"]
        data, metadatas = self.data_retriever.retrieve_data(
                                                            site_latitude=site_latitude,
                                                            site_longitude=site_longitude,
                                                            time_of_interest=time_of_interest
                                                            )
        # No data found during the specified time range
        if data is None:
            raise DataNotFoundException("No data found.")
        return {"data": data, "metadatas": metadatas, "message": "Data retrieved successfully."}