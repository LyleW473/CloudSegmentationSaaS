import set_path
import pytest
import subprocess
import random

from fastapi import status
from src.backend.settings.paths import ENDPOINT_URLS
from src.testing.utils import make_request, check_query_result

user_authentication_url = ENDPOINT_URLS["web_app"]["base_url"] + ENDPOINT_URLS["web_app"]["additional_paths"]["user_authentication"]

def generate_random_username() -> str:
    """
    Generates a random username for testing.

    Returns:
        str: The generated username.
    """
    return f"test_{random.randint(0, 99999)}"

test_username = generate_random_username()
correct_password = "StrongPassword123!"
incorrect_password = "RandomPassword123!"

queries = {
            "weak_password": {
                            "username": generate_random_username(),
                            "password": "weakpassword"
                            },
            "no_uppercase": {
                            "username": generate_random_username(),
                            "password": "weakpassword123!"
                            },
            "no_lowercase": {
                            "username": generate_random_username(),
                            "password": "WEAKPASSWORD123!"
                            },
            "no_number": {
                            "username": generate_random_username(),
                            "password": "WeakPassword!"
                            },
            "no_special_character": {
                            "username": generate_random_username(),
                            "password": "WeakPassword123"
                            },
}

expected_status_codes_map = {query_type: status.HTTP_400_BAD_REQUEST for query_type in queries.keys()}

@pytest.mark.order(1)
def test_signup(app_server:subprocess.Popen) -> None:
    """
    Tests that a user can sign up successfully with a 
    valid username and a strong password.

    Args:
        app_server (subprocess.Popen): The server process that is running the web app and its endpoints.
    """
    data = {"username": test_username, "password": correct_password}
    response = make_request("POST", user_authentication_url, json=data)
    assert response.status_code == status.HTTP_201_CREATED

@pytest.mark.order(2)
def test_incorrect_password(app_server:subprocess.Popen) -> None:
    """
    Tests that a user cannot log into the system with an incorrect password.

    Args:
        app_server (subprocess.Popen): The server process that is running the web app and its endpoints.
    """
    data = {"username": test_username, "password": incorrect_password}
    response = make_request("POST", user_authentication_url, json=data)
    assert response.status_code == status.HTTP_401_UNAUTHORIZED

@pytest.mark.order(3)
def test_correct_password(app_server:subprocess.Popen) -> None:
    """
    Tests that a user can log into the system with a correct password.

    Args:
        app_server (subprocess.Popen): The server process that is running the web app and its endpoints.
    """
    data = {"username": test_username, "password": correct_password}
    response = make_request("POST", user_authentication_url, json=data)
    assert response.status_code == status.HTTP_200_OK

@pytest.mark.order(4)
@pytest.mark.parametrize("query_type", queries.keys())
def test_all_queries(app_server:subprocess.Popen, query_type:str) -> None:
    """
    Tests whether the user can/cannot sign up with each query.
    E.g.
    - Weak password
    - No uppercase letter
    - No lowercase letter
    - No number
    - No special character

    Args:
        app_server (subprocess.Popen): The server process that is running the web app and its endpoints.
        query_type (str): The type of query to test, e.g., "weak_password".
    """
    check_query_result(
                    user_authentication_url=user_authentication_url,
                    request_data=queries[query_type],
                    expected_status_code=expected_status_codes_map[query_type],
                    endpoint_url=user_authentication_url
                    )