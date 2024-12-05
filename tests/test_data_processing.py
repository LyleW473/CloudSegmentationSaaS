import set_path
import pytest
import subprocess
import numpy as np
from fastapi import status

from src.backend.settings.paths import ENDPOINT_URLS
from src.testing.utils import make_request, check_query_result

# Define queries for the test cases
patch_size = 384
red_band = np.random.randn(patch_size, patch_size).tolist()
green_band = np.random.randn(patch_size, patch_size).tolist()
blue_band = np.random.randn(patch_size, patch_size).tolist()
nir08_band = np.random.randn(patch_size, patch_size).tolist()
random_3d_array = np.random.randn(patch_size, patch_size, 4).tolist()

queries = {
            "valid_query": {
                            "data":
                                {
                                "red": red_band,
                                "green":green_band,
                                "blue": blue_band,
                                "nir08": nir08_band,
                                }
                            },
            "missing_data_key": { 
                                "red": red_band,
                                "green":green_band,
                                "blue": blue_band,
                                "nir08": nir08_band,
                                },
            "missing_red_band": {
                                "data":
                                    {   
                                    "green":green_band,
                                    "blue": blue_band,
                                    "nir08": nir08_band,
                                    }
                                },
            "missing_green_band": {
                                "data":
                                    {
                                    "red": red_band,
                                    "blue": blue_band,
                                    "nir08": nir08_band,
                                    }
                                },
            "missing_blue_band": {
                                "data":
                                    {
                                    "red": red_band,
                                    "green":green_band,
                                    "nir08": nir08_band,
                                    }
                                },
            "invalid_band_shape": {
                                "data":
                                    {
                                    "red": red_band,
                                    "green":green_band,
                                    "blue": blue_band,
                                    "nir08": random_3d_array,
                                    }
                                }
            }
expected_status_codes_map = {
                            "valid_query": status.HTTP_200_OK,
                            "missing_data_key": status.HTTP_400_BAD_REQUEST,
                            "missing_red_band": status.HTTP_400_BAD_REQUEST,
                            "missing_green_band": status.HTTP_400_BAD_REQUEST,
                            "missing_blue_band": status.HTTP_400_BAD_REQUEST,
                            "invalid_band_shape": status.HTTP_400_BAD_REQUEST,
                            }

data_processing_url = ENDPOINT_URLS["data_processing"]["base_url"] + ENDPOINT_URLS["data_processing"]["path"]
user_authentication_url = ENDPOINT_URLS["web_app"]["base_url"] + ENDPOINT_URLS["web_app"]["additional_paths"]["user_authentication"]

@pytest.mark.order(1)
def test_unauthenticated_request(app_server:subprocess.Popen) -> None:
    """
    Tests that the user cannot process their data without being authenticated.

    Args:
        app_server (subprocess.Popen): The server process that is running the web app and its endpoints.
    """
    data = queries["valid_query"]
    response = make_request("POST", data_processing_url, json=data)
    assert response.status_code == status.HTTP_401_UNAUTHORIZED

@pytest.mark.order(2)
@pytest.mark.parametrize("query_type", list(queries.keys()))
def test_all_queries(app_server:subprocess.Popen, query_type:str) -> None:
    """
    Tests whether the user can/cannot process their data with each query.
    E.g:
    - Valid query
    - Missing 'data' key
    - Missing 'red' band
    - Missing 'green' band
    - Missing 'blue' band
    - Invalid band shape
    - (And more logic-based tests which reside inside the endpoint itself)

    Args:
        app_server (subprocess.Popen): The server process that is running the web app and its endpoints.
        query_type (str): The type of query to test, e.g., "valid_query"
    """
    check_query_result(
                    user_authentication_url=user_authentication_url,
                    request_data=queries[query_type],
                    expected_status_code=expected_status_codes_map[query_type],
                    endpoint_url=data_processing_url
                    )