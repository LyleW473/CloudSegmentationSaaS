import set_path
import pytest
import subprocess
from fastapi import status

from src.backend.settings.paths import ENDPOINT_URLS
from src.testing.utils import make_request, check_query_result

valid_date_range = "2021-01-01/2024-01-02"
invalid_date_range = "2021-01-01/2020-01-02"
test_locations = {
                "Statue of Liberty": (40.6892, -74.0445, valid_date_range, status.HTTP_200_OK), 
                "River Thames": (51.5072, 0.1276, valid_date_range, status.HTTP_200_OK),
                "Invalid location (1)": (200, -74.0445, valid_date_range, status.HTTP_400_BAD_REQUEST),
                "Invalid location (2)": (40.6892, -200, valid_date_range, status.HTTP_400_BAD_REQUEST),
                "Invalid time range": (40.6892, -74.0445, invalid_date_range, status.HTTP_400_BAD_REQUEST)
                }

# Build the queries and expected status codes for each test location
queries = {}
status_codes_for_locations = {}
for location, (latitude, longitude, time_range, expected_status_code) in test_locations.items():
    queries[location] = {
                        "site_latitude": latitude,
                        "site_longitude": longitude,
                        "time_of_interest": time_range
                        }
    status_codes_for_locations[location] = expected_status_code

data_retrieval_url = ENDPOINT_URLS["data_retrieval"]["base_url"] + ENDPOINT_URLS["data_retrieval"]["path"]
user_authentication_url = ENDPOINT_URLS["web_app"]["base_url"] + ENDPOINT_URLS["web_app"]["additional_paths"]["user_authentication"]

@pytest.mark.order(1)
@pytest.mark.parametrize("endpoint", [data_retrieval_url])
@pytest.mark.parametrize("query_type", list(test_locations.keys()))
def test_unauthenticated_request(app_server:subprocess.Popen, endpoint:str, query_type:str) -> None:
    """
    Tests that the user cannot retrieve data without being authenticated.

    Args:
        app_server (subprocess.Popen): The server process that is running the web app and its endpoints.
        endpoint (str): The URL of the endpoint to test.
        query_type (str): The type of query to test, e.g., "Statue of Liberty"
    """
    data = queries[query_type]
    response = make_request("POST", endpoint, json=data)
    assert response.status_code == status.HTTP_401_UNAUTHORIZED

@pytest.mark.order(2)
@pytest.mark.parametrize("endpoint", [data_retrieval_url])
@pytest.mark.parametrize("query_type", list(test_locations.keys()))
def test_all_locations(app_server:subprocess.Popen, endpoint:str, query_type:str) -> None:
    """
    Tests whether the user can/cannot retrieve data for each location.
    E.g:
    - Valid locations
    - Longitude is not in range [-180, 180]
    - Latitude is not in range [-90, 90]
    - The start date is after the end date
    - (And more logic-based tests which reside inside the endpoint itself)

    Args:
        app_server (subprocess.Popen): The server process that is running the web app and its endpoints.
        endpoint (str): The URL of the endpoint to test.
        query_type (str): The type of query to test, e.g., "Statue of Liberty"
    """
    check_query_result(
                        user_authentication_url=user_authentication_url,
                        request_data=queries[query_type],
                        expected_status_code=status_codes_for_locations[query_type],
                        endpoint_url=endpoint
                        )