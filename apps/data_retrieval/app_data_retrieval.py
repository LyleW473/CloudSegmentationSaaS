if __name__ == "__main__":
    import set_path
import logging
from fastapi import FastAPI, HTTPException, Request, Body, Depends
from fastapi.responses import JSONResponse
from fastapi import status
from fastapi.exceptions import RequestValidationError

from src.backend.data_retrieval.retrieval_pipeline import RetrievalPipeline
from src.backend.utils.exceptions import InvalidQueryException, DataNotFoundException
from src.backend.utils.pydantic_models import SiteLocationQuery
from src.backend.settings.paths import ENDPOINT_URLS
from src.backend.user_authentication.utils import validate_request

app = FastAPI()
retrieval_pipeline = RetrievalPipeline()
log = logging.getLogger(__name__)

@app.exception_handler(RequestValidationError)
async def input_validation_exception_handler(request:Request, exc:RequestValidationError):
    message = (
            "Invalid input parameters."
            " Please provide a JSON containing 'site_latitude', 'site_longitude', and 'time_of_interest' keys."
            " 'site_latitude' should be a float or int in the range [-90, 90]."
            " 'site_longitude' should be a float or int in the range [-180, 180]."
            " 'time_of_interest' should be a string in the format 'YYYY-MM-DD/YYYY-MM-DD'."
            )
    return JSONResponse(
                        content={
                                "data": None, 
                                "message": message
                                }, 
                        status_code=status.HTTP_400_BAD_REQUEST
                        )

@app.post(
        ENDPOINT_URLS['data_retrieval']['path'], 
        response_class=JSONResponse, 
        dependencies=[Depends(validate_request)]
        )
async def root(query:SiteLocationQuery=Body(...)) -> JSONResponse:
    """
    Server-side function for performing model inference on a specified image file.
    - Accepts a JSON body containing the query parameters.
    
    Args:
        query (SiteLocationQuery): The dictionary of input parameters. In this case, 
                                   the site location and time of interest.
    """
    try:
        result = retrieval_pipeline(query.model_dump())
        return JSONResponse(content=result, status_code=status.HTTP_200_OK)
    except Exception as e:
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        if isinstance(e, InvalidQueryException):
            status_code = status.HTTP_400_BAD_REQUEST
        elif isinstance(e, DataNotFoundException):
            status_code = status.HTTP_404_NOT_FOUND
        elif isinstance(e, HTTPException):
            status_code = e.status_code
        log.error(f"An error occurred in the data retrieval service. Error: {str(e)}")
        return JSONResponse(content={"data": None, "message": str(e)}, status_code=status_code)
    
if __name__ == "__main__":
    import uvicorn
    from src.backend.settings.paths import ENDPOINT_URLS

    app_name = ENDPOINT_URLS['data_retrieval']['app_name']
    base_url = ENDPOINT_URLS['data_retrieval']['base_url']
    url = base_url.split("http://")[-1]
    _, port = url.split(":")
    uvicorn.run(f"{app_name}:app", host="0.0.0.0", port=int(port))