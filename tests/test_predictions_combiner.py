import set_path
import pytest
import subprocess
import numpy as np

from fastapi import status
from src.backend.settings.paths import ENDPOINT_URLS
from src.testing.utils import make_request, check_query_result

num_patches = 16
patch_size = 384
stride = patch_size // 2
original_img_size = (patch_size * 2, patch_size * 2) # (Just an example)
valid_array = np.random.randn(num_patches, patch_size, patch_size).tolist()
too_many_dimensions_array_1 = np.random.randn(num_patches, patch_size, patch_size, 1).tolist()
too_many_dimensions_array_2 = np.random.randn(num_patches, 1, patch_size, patch_size).tolist()
invalid_order_array = np.random.randn(patch_size, patch_size, num_patches).tolist()

queries = {
            "valid_query": {
                            "data": valid_array,
                            "stride": stride,
                            "patch_size": patch_size,
                            "original_img_height": original_img_size[0],
                            "original_img_width": original_img_size[1]
                            },
            "missing_data_key": {
                                "stride": stride,
                                "patch_size": patch_size,
                                "original_img_height": original_img_size[0],
                                "original_img_width": original_img_size[1]
                                },
            "missing_stride_key": {
                                "data": valid_array,
                                "patch_size": patch_size,
                                "original_img_height": original_img_size[0],
                                "original_img_width": original_img_size[1]
                                },
            "missing_patch_size_key": {
                                "data": valid_array,
                                "stride": stride,
                                "original_img_height": original_img_size[0],
                                "original_img_width": original_img_size[1]
                                },
            "missing_original_img_height_key": {
                                "data": valid_array,
                                "stride": stride,
                                "patch_size": patch_size,
                                "original_img_width": original_img_size[1]
                                },
            "missing_original_img_width_key": {
                                "data": valid_array,
                                "stride": stride,
                                "patch_size": patch_size,
                                "original_img_height": original_img_size[0]
                                },
            "invalid_data_key_1": {
                                "data": too_many_dimensions_array_1,
                                "stride": stride,
                                "patch_size": patch_size,
                                "original_img_height": original_img_size[0],
                                "original_img_width": original_img_size[1]
                                },
            "invalid_data_key_2": {
                                "data": too_many_dimensions_array_2,
                                "stride": stride,
                                "patch_size": patch_size,
                                "original_img_height": original_img_size[0],
                                "original_img_width": original_img_size[1]
                                },
            "invalid_data_key_3": {
                                "data": invalid_order_array,
                                "stride": stride,
                                "patch_size": patch_size,
                                "original_img_height": original_img_size[0],
                                "original_img_width": original_img_size[1]
                                }
            }
expected_status_codes_map = {
                            "valid_query": status.HTTP_200_OK,
                            "missing_data_key": status.HTTP_400_BAD_REQUEST,
                            "missing_stride_key": status.HTTP_400_BAD_REQUEST,
                            "missing_patch_size_key": status.HTTP_400_BAD_REQUEST,
                            "missing_original_img_height_key": status.HTTP_400_BAD_REQUEST,
                            "missing_original_img_width_key": status.HTTP_400_BAD_REQUEST,
                            "invalid_data_key_1": status.HTTP_400_BAD_REQUEST,
                            "invalid_data_key_2": status.HTTP_400_BAD_REQUEST,
                            "invalid_data_key_3": status.HTTP_422_UNPROCESSABLE_ENTITY # Unprocessable because of the order of the dimensions
                            }

user_authentication_url = ENDPOINT_URLS["web_app"]["base_url"] + ENDPOINT_URLS["web_app"]["additional_paths"]["user_authentication"]
predictions_combiner_url = ENDPOINT_URLS["predictions_combiner"]["base_url"] + ENDPOINT_URLS["predictions_combiner"]["path"]

@pytest.mark.order(1)
@pytest.mark.parametrize("query_type", list(queries.keys()))
def test_unauthenticated_request(app_server:subprocess.Popen, query_type:str) -> None:
    """
    Tests that the user cannot perform predictions combining without being authenticated.

    Args:
        app_server (subprocess.Popen): The server process that is running the web app and its endpoints.
        query_type (str): The type of query to test, e.g., "valid_query".
    """
    data = queries[query_type]
    response = make_request("POST", predictions_combiner_url, json=data)
    assert response.status_code == status.HTTP_401_UNAUTHORIZED

@pytest.mark.order(2)
@pytest.mark.parametrize("query_type", list(queries.keys()))
def test_all_queries(app_server:subprocess.Popen, query_type:str) -> None:
    """
    Tests whether the user can/cannot perform predictions combining with each query.
    E.g.
    - Valid query
    - Missing "data" key
    - Missing "stride" key
    - Missing "patch_size" key
    - Missing "original_img_height" key
    - Missing "original_img_width" key
    - Invalid query with too many dimensions
    - Invalid query with the wrong order of dimensions

    Args:
        app_server (subprocess.Popen): The server process that is running the web app and its endpoints.
        query_type (str): The type of query to test, e.g., "valid_query".
    """
    check_query_result(
                    user_authentication_url=user_authentication_url,
                    request_data=queries[query_type],
                    expected_status_code=expected_status_codes_map[query_type],
                    endpoint_url=predictions_combiner_url
                    )