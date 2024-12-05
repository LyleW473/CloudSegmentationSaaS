if __name__ == "__main__":
    import set_path

import logging.config
import os
import logging

from fastapi import FastAPI, Request, Body, Depends
from fastapi import status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from src.backend.settings.paths import MAIN_SAVED_IMAGES_DIR, ENDPOINT_URLS, TEMPLATES_DIR
from src.backend.settings.logging_config import LOGGING_CONFIG
from src.backend.utils.pydantic_models import SiteLocationQuery
from src.backend.user_authentication.utils import validate_request
from src.backend.user_authentication.authentication_service import UserAuthenticationService
from src.backend.coordinator.query_submitter import QuerySubmitter
from src.frontend.utils import format_results

app = FastAPI()
logging.config.dictConfig(LOGGING_CONFIG)
log = logging.getLogger(__name__)
log.info("Frontend server started")

# Mount the saved_images directory to the /saved_images endpoint (for serving the saved images)
if not os.path.exists(MAIN_SAVED_IMAGES_DIR):
    os.makedirs(MAIN_SAVED_IMAGES_DIR)
    os.makedirs(f"{MAIN_SAVED_IMAGES_DIR}/images")
    os.makedirs(f"{MAIN_SAVED_IMAGES_DIR}/masks")
app.mount(
	f"/{MAIN_SAVED_IMAGES_DIR}", 
	StaticFiles(directory=MAIN_SAVED_IMAGES_DIR), 
	name=MAIN_SAVED_IMAGES_DIR
	)

# Create a new client and connect to the server
load_dotenv()
is_testing = os.getenv("IS_TESTING", "false").lower() == "true"
print(is_testing)
user_authentication_service = UserAuthenticationService(is_testing=is_testing)
query_submitter = QuerySubmitter()

# Add CORS middleware to allow requests from the frontend (localhost)
origins = [
            "http://localhost",
            "http://localhost:8080",
            "http://127.0.0.1",
            ]

app.add_middleware(
                    CORSMiddleware,
                    allow_origins=origins,
                    allow_credentials=True, # Allows cookies to be sent to the frontend, so that they can make authenticated requests
                    allow_methods=["GET", "POST", "OPTIONS"],
                    allow_headers=["Content-Type", "Authorization"],
                    )

templates = Jinja2Templates(directory=TEMPLATES_DIR)

@app.get(ENDPOINT_URLS['web_app']['path'], response_class=HTMLResponse, dependencies=[Depends(validate_request)])
async def root(request:Request):
    return templates.TemplateResponse(
                                    "index.html", 
                                    {"request": request}
                                    )

@app.post(ENDPOINT_URLS['web_app']['additional_paths']['user_authentication'], response_class=JSONResponse)
async def user_authentication(request:Request, username:str=Body(...), password:str=Body(...)) -> JSONResponse:
    """
    Authenticates the user by checking the username and password provided.
    - If the user is authenticated, a token is generated and set in the cookie.
    - The token is used for making authenticated requests to the rest of the system.

    Args:
        request (Request): The request object containing information
                           that can be used to authenticate the user.
        username (str): The username of the user.
        password (str): The password of the user.

    Returns:
        JSONResponse: A JSON response containing the authentication token if the user is authenticated.
    """
    log.info(f"User authentication request received for username '{username}' ...")
    is_rate_limited, message = user_authentication_service.handle_rate_limiting(request=request, username=username)
    if is_rate_limited:
        return JSONResponse(content={"message": message}, status_code=status.HTTP_429_TOO_MANY_REQUESTS)
    
    status_code, message = user_authentication_service.handle_authentication(username=username, password=password, request=request)
    if not (status_code == status.HTTP_200_OK or status_code == status.HTTP_201_CREATED):
        return JSONResponse(content={"message": message}, status_code=status_code)
    log.info("Successfully authenticated user ...")
    return user_authentication_service.get_token_response(username=username, status_code=status_code, message=message)

@app.post(
        ENDPOINT_URLS['web_app']['additional_paths']['query'], 
        response_class=JSONResponse, 
        dependencies=[Depends(validate_request)]
        )
async def submit_query(request:Request, query:SiteLocationQuery=Body(...)) -> JSONResponse:
    """
    Sends a query to the pipeline coordinator to process the site location and time of interest.
    - The query should contain the site location and time of interest.

    Args:
        request (Request): The request object containing information that can be used to authenticate the 
                           user.
        query (SiteLocationQuery): The query object containing the site location and time of interest.
    """
    try:    
        log.info("Submitting query to the system ...")
        status_code, content = query_submitter.submit(request=request, query=query)
        return JSONResponse(content=content, status_code=status_code)
    except Exception as e:
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        message = f"An error occurred in the web-app service. Error: {str(e)}"
        log.error(message)
        return JSONResponse(content={"message": message, "data": None}, status_code=status_code)
    
@app.get(ENDPOINT_URLS['web_app']['additional_paths']['login'], response_class=HTMLResponse)
async def login(request:Request) -> HTMLResponse:
    """
    Displays the login page.

    Args:
        request (Request): The request object containing information
                           that can be used/displayed in the template.
    """
    return templates.TemplateResponse("login.html", {"request": request})

@app.get(
        ENDPOINT_URLS['web_app']['additional_paths']['result'], 
        response_class=HTMLResponse, 
        dependencies=[Depends(validate_request)]
        )
async def result(request:Request) -> HTMLResponse:
    """
    Displays the results of the query in an HTML template.

    Args:
        request (Request): The request object containing information
                           that can be used/displayed in the template.
    """
    query_params = ["site_latitude", "site_longitude", "time_of_interest", "time_taken"]
    query = {param: request.query_params.get(param) for param in query_params}
    results = format_results()
    return templates.TemplateResponse(
                                    "result.html", 
                                    {
                                    "request": request,
                                    "query": query,
                                    "results": results,
                                    "time_taken": query["time_taken"]
                                    }
                                    )

if __name__ == "__main__":
    import uvicorn
    from src.backend.settings.paths import ENDPOINT_URLS

    app_name = ENDPOINT_URLS['web_app']['app_name']
    base_url = ENDPOINT_URLS['web_app']['base_url']
    url = base_url.split("http://")[-1]
    _, port = url.split(":")
    uvicorn.run(f"{app_name}:app", host="0.0.0.0", port=int(port))