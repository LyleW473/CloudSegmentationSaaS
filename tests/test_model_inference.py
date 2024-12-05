import set_path
import pytest
import subprocess
import numpy as np
from fastapi import status

from src.backend.settings.paths import ENDPOINT_URLS
from src.testing.utils import make_request, check_query_result

four_d_image_array = np.random.randn(8, 384, 384, 4).tolist()
four_d_image_array_2 = np.random.randn(1, 384, 384, 4).tolist()
three_d_image_array = np.random.randn(384, 384, 4).tolist()


user_authentication_url = ENDPOINT_URLS["web_app"]["base_url"] + ENDPOINT_URLS["web_app"]["additional_paths"]["user_authentication"]
model_inference_url = ENDPOINT_URLS["model_inference"]["base_url"] + ENDPOINT_URLS["model_inference"]["path"]

queries = {
            "valid_query": {
                            "data": four_d_image_array
                            },
            "valid_query_2": {
                            "data": four_d_image_array_2
                            },
            "valid_query_3": {
                            "data": four_d_image_array,
                            "some_key": "some_value"
                            },
            "invalid_query_1": {
                            "data": three_d_image_array
                            },
            "invalid_query_2": {
                            "some_key": four_d_image_array
                            },
            "invalid_query_3": {
                            }
            }

expected_status_codes_map = {
                            "valid_query": status.HTTP_200_OK,
                            "valid_query_2": status.HTTP_200_OK,
                            "valid_query_3": status.HTTP_200_OK,
                            "invalid_query_1": status.HTTP_400_BAD_REQUEST,
                            "invalid_query_2": status.HTTP_400_BAD_REQUEST,
                            "invalid_query_3": status.HTTP_400_BAD_REQUEST
                            }

@pytest.mark.order(1)
def test_unauthenticated_request(app_server:subprocess.Popen) -> None:
    """
    Tests that the user cannot perform model inference without being authenticated.

    Args:
        app_server (subprocess.Popen): The server process that is running the web app and its endpoints.
    """
    request_data = {"data": four_d_image_array}
    response = make_request("POST", model_inference_url, json=request_data)
    assert response.status_code == status.HTTP_401_UNAUTHORIZED

@pytest.mark.order(2)
@pytest.mark.parametrize("query_type", list(queries.keys()))
def test_all_queries(app_server:subprocess.Popen, query_type:str) -> None:
    """
    Tests whether the user can/cannot perform model inference with each query.
    E.g.
    - Valid query
    - Valid query with an extra (unnecessary) key
    - Invalid query with a 3D image array
    - Invalid query with a missing "data" key
    - Invalid query with an empty dictionary
    
    Args:
        app_server (subprocess.Popen): The server process that is running the web app and its endpoints.
        query_type (str): The type of query to test, e.g., "valid_query".
    """
    check_query_result(
                        user_authentication_url=user_authentication_url,
                        request_data=queries[query_type],
                        expected_status_code=expected_status_codes_map[query_type],
                        endpoint_url=model_inference_url
                        )