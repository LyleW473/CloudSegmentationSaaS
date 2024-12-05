if __name__ == "__main__":
    import set_path
import logging
from fastapi import FastAPI, HTTPException, Request, Body, Depends
from fastapi.responses import JSONResponse
from fastapi import status
from fastapi.exceptions import RequestValidationError

from src.backend.model_inference.inference_pipeline import InferencePipeline
from src.backend.utils.exceptions import InvalidQueryException, ModelLoadingException, ModelInferenceException
from src.backend.utils.pydantic_models import PatchImagesQuery
from src.backend.settings.paths import ENDPOINT_URLS
from src.backend.user_authentication.utils import validate_request

app = FastAPI()
inference_pipeline = InferencePipeline()
log = logging.getLogger(__name__)

@app.exception_handler(RequestValidationError)
async def input_validation_exception_handler(request:Request, exc:RequestValidationError):
    message = (
            "Invalid input parameters."
            " Please provide a JSON containing the 'data' key."
            " 'data' should be a 4-D image array of shape (num_patches, height, width, num_channels) represented as a list of lists."
            )
    return JSONResponse(
                        content={
                                "data": None, 
                                "message": message
                                }, 
                        status_code=status.HTTP_400_BAD_REQUEST
                        )
@app.post(
        ENDPOINT_URLS['model_inference']['path'], 
        response_class=JSONResponse, 
        dependencies=[Depends(validate_request)]
        )
async def root(query:PatchImagesQuery=Body(...)) -> JSONResponse:
    """
    Server-side function for performing model inference on image patches.
    
    Args:
        query (PatchImagesQuery): The dictionary of input parameters.
    """
    try:
        result = inference_pipeline(query.model_dump())
        return JSONResponse(content=result, status_code=status.HTTP_200_OK)
    except Exception as e:
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        if isinstance(e, InvalidQueryException):
            status_code = status.HTTP_400_BAD_REQUEST
        elif isinstance(e, ModelLoadingException):
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        elif isinstance(e, ModelInferenceException):
            status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
        elif isinstance(e, HTTPException):
            status_code = e.status_code
        log.error(f"An error occurred in the model inference service. Error: {str(e)}")
        return JSONResponse(content={"data": None, "message": str(e)}, status_code=status_code)
    
if __name__ == "__main__":
    import uvicorn
    from src.backend.settings.paths import ENDPOINT_URLS

    app_name = ENDPOINT_URLS['model_inference']['app_name']
    base_url = ENDPOINT_URLS['model_inference']['base_url']
    url = base_url.split("http://")[-1]
    _, port = url.split(":")
    uvicorn.run(f"{app_name}:app", host="0.0.0.0", port=int(port))