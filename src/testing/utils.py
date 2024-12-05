import requests

from typing import Any

def make_request(method, url, **kwargs) -> requests.Response:
    """
    Helper funcgtion to make a request to a specified URL
    with the specified method and keyword arguments.
    """
    return requests.request(method, url, **kwargs)

def retrieve_user_authentication_token(user_authentication_url:str) -> str:
    """
    Helper function to retrieve the user authentication token from the web app
    to allow the user to access the endpoints.

    Args:
        user_authentication_url (str): The URL of the user authentication endpoint.
    """
    data = {"username": "test_123", "password": "StrongPassword123!"}
    response = make_request("POST", user_authentication_url, json=data)
    token = response.json()["token"]
    return token

def check_query_result(user_authentication_url:str, request_data:Any, expected_status_code:int, endpoint_url:str) -> None:
    """
    Helper function to check the result of a query to the specified endpoint
    with the expected status code.
    
    Args:
        user_authentication_url (str): The URL of the user authentication endpoint
        request_data (Any): The data to be sent in the request.
        expected_status_code (int): The expected status code of the response.
        endpoint_url (str): The URL of the endpoint to test.
    """
    # Get a token to authenticate the user
    token = retrieve_user_authentication_token(user_authentication_url)

    # Create a request to the data retrieval endpoint
    # print("TOKEN", token)
    headers = {"Authorization": token}
    response = make_request("POST", endpoint_url, json=request_data, headers=headers)
    
    # print("Response Status Code:", response.status_code)
    # print("Expected Status Code:", expected_status_code)
    assert response.status_code == expected_status_code