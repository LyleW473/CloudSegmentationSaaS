"""
This file contains the fixtures that are used in the tests.
They are automatically detected and used by pytest when the tests are run.
"""

import pytest
import subprocess
import sys
import time
import os
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv

@pytest.fixture(scope="session")
def app_server():
    """
    Fixture to start the server before running the tests and 
    close it after the tests are done.
    """

    # Start the server and run it
    server = subprocess.Popen([sys.executable, "scripts/run_apps.py"])

    # Wait for the server to start
    time.sleep(10)

    try:
        yield server
    finally:
        # Close the server
        server.kill()
        server.wait()

@pytest.fixture(scope="session", autouse=True)
def reset_test_database(app_server:subprocess.Popen):
    """
    Fixture to reset the test database before and after running 
    the tests.

    - The 'app_server' fixture is used to ensure that the server
      is running before the database is reset.
    """
    load_dotenv()
    uri = os.getenv("MONGODB_URI")
    client = MongoClient(
                        uri, 
                        server_api=ServerApi('1'),
                        connectTimeoutMS=60000, 
                        socketTimeOutMS=60000
                        )
    db = client["user_authentication"]

    # Drop the 'test_users' collection before running the tests
    db.drop_collection("test_users")

    yield # Run the tests

    # Drop the 'test_users' collection after the tests are done
    db.drop_collection("test_users")

    client.close()