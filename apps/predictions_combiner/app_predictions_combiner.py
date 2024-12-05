if __name__ == "__main__":
	import set_path
import logging
from fastapi import FastAPI, HTTPException, Request, Body, Depends
from fastapi.responses import JSONResponse
from fastapi import status
from fastapi.exceptions import RequestValidationError

from src.backend.combiner.combiner_pipeline import CombinerPipeline
from src.backend.utils.exceptions import InvalidQueryException, PredictionsCombiningException
from src.backend.utils.pydantic_models import PatchSegmentationMasksQuery
from src.backend.settings.paths import ENDPOINT_URLS
from src.backend.user_authentication.utils import validate_request

app = FastAPI()
combiner_pipeline = CombinerPipeline()
log = logging.getLogger(__name__)

@app.exception_handler(RequestValidationError)
async def input_validation_exception_handler(request:Request, exc:RequestValidationError):
    message = (
            "Invalid input parameters."
            " Please provide a JSON containing the following keys: "
            "'data': A 3-D image array of shape (num_patches, height, width) represented as a list of lists."
            "'stride': The stride used to extract the patches."
            "'patch_size': The size of the patches extracted."
            "'original_img_height': The height of the original image."
            "'original_img_width': The width of the original image."
            )
    return JSONResponse(
                        content={
                                "data": None, 
                                "message": message
                                }, 
                        status_code=status.HTTP_400_BAD_REQUEST
                        )

@app.post(
        ENDPOINT_URLS['predictions_combiner']['path'], 
        response_class=JSONResponse, 
        dependencies=[Depends(validate_request)]
        )
async def root(query:PatchSegmentationMasksQuery=Body(...)) -> JSONResponse:
    """
    Server-side function for combining the segmentation masks of image patches
    into a single segmentation mask.
    
    Args:
        query (PatchSegmentationMasksQuery): The dictionary of input parameters.
    """
    try:
        result = combiner_pipeline(query.model_dump())
        return JSONResponse(content=result, status_code=status.HTTP_200_OK)
    except Exception as e:
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        if isinstance(e, InvalidQueryException):
            status_code = status.HTTP_400_BAD_REQUEST
        elif isinstance(e, PredictionsCombiningException):
            status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
        elif isinstance(e, HTTPException):
            status_code = e.status_code
        log.error(f"An error occurred in the predictions combiner service. Error: {str(e)}")
        return JSONResponse(content={"data": None, "message": str(e)}, status_code=status_code)

if __name__ == "__main__":
    import uvicorn
    from src.backend.settings.paths import ENDPOINT_URLS

    app_name = ENDPOINT_URLS['predictions_combiner']['app_name']
    base_url = ENDPOINT_URLS['predictions_combiner']['base_url']
    url = base_url.split("http://")[-1]
    _, port = url.split(":")
    uvicorn.run(f"{app_name}:app", host="0.0.0.0", port=int(port))