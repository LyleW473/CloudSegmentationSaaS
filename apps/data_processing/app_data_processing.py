if __name__ == "__main__":
    import set_path
import logging
from fastapi import FastAPI, HTTPException, Request, Body, Depends
from fastapi.responses import JSONResponse
from fastapi import status
from fastapi.exceptions import RequestValidationError

from src.backend.data_processing import ProcessingPipeline
from src.backend.utils.exceptions import InvalidQueryException
from src.backend.utils.pydantic_models import ExtractedBandsQuery
from src.backend.settings.paths import ENDPOINT_URLS
from src.backend.user_authentication.utils import validate_request

app = FastAPI()
processing_pipeline = ProcessingPipeline()
log = logging.getLogger(__name__)

@app.exception_handler(RequestValidationError)
async def input_validation_exception_handler(request:Request, exc:RequestValidationError):
    message = (
            "Invalid input parameters."
            " Please provide a JSON with a 'data' key that maps strings to (height, width) arrays of type float or int."
            " The 'data' key should contain the keys: 'red', 'green', 'blue', 'nir08'."
            )
    return JSONResponse(
                        content={
                                "data": None, 
                                "message": message
                                }, 
                        status_code=status.HTTP_400_BAD_REQUEST
                        )
@app.post(
        ENDPOINT_URLS['data_processing']['path'], 
        response_class=JSONResponse, 
        dependencies=[Depends(validate_request)]
        )
async def root(query:ExtractedBandsQuery=Body(...)) -> JSONResponse:
    """
    Server-side function for performing model inference on a specified image file.
    
    Args:
        query (ExtractedBandsQuery): The dictionary of input parameters.
    """
    try:
        result = processing_pipeline(query.model_dump())
        return JSONResponse(content=result, status_code=status.HTTP_200_OK)
    except Exception as e:
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        if isinstance(e, InvalidQueryException):
            status_code = status.HTTP_400_BAD_REQUEST
        elif isinstance(e, HTTPException):
            status_code = e.status_code
        log.error(f"An error occurred in the data processing service. Error: {str(e)}")
        return JSONResponse(content={"data": None, "message": str(e)}, status_code=status_code)
    
if __name__ == "__main__":
    import uvicorn
    from src.backend.settings.paths import ENDPOINT_URLS

    app_name = ENDPOINT_URLS['data_processing']['app_name']
    base_url = ENDPOINT_URLS['data_processing']['base_url']
    url = base_url.split("http://")[-1]
    _, port = url.split(":")
    uvicorn.run(f"{app_name}:app", host="0.0.0.0", port=int(port))