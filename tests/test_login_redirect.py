import set_path
import pytest
import subprocess
from fastapi import status

from src.backend.settings.paths import ENDPOINT_URLS
from src.testing.utils import make_request

@pytest.mark.order(1)
def test_redirect_to_login(app_server:subprocess.Popen) -> None:
    """
    Test that the user is redirected to the login page when they visit the main page
    without being logged in / authenticated.

    Args:
        app_server (subprocess.Popen): The server process that is running the web app and its endpoints.
    """
    main_home_page_url = ENDPOINT_URLS["web_app"]["base_url"]
    login_page_url = main_home_page_url + ENDPOINT_URLS["web_app"]["additional_paths"]["login"]
    response = make_request("GET", main_home_page_url)

    redirect_response = response.history[-1] # Should be the redirect response
    assert response.url == login_page_url
    assert response.status_code == status.HTTP_200_OK
    assert redirect_response.status_code == status.HTTP_302_FOUND