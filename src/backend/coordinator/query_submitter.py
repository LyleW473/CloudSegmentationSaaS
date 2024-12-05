import logging

from fastapi import Request
from fastapi import status
from src.backend.coordinator.coordinator import PipelineCoordinator
from src.backend.settings.paths import ENDPOINT_URLS
from src.backend.utils.pydantic_models import SiteLocationQuery
from src.frontend.utils import save_results

class QuerySubmitter:
    """
    Class responsible for submitting the query entered by the user to the pipeline coordinator,
    returning the predicted segmentation masks, any intermediate results and more.
    """
    def __init__(self):
        """
        Initializes the QuerySubmitter object.
        - Creates a pipeline coordinator, pasing the URLs of the services that the coordinator will call.
        """
        pipeline_urls = {key: f"{ENDPOINT_URLS[key]['base_url']}{ENDPOINT_URLS[key]['path']}" for key in ENDPOINT_URLS}
        self.pipeline_coordinator = PipelineCoordinator(pipeline_urls)
        self.log = logging.getLogger(self.__class__.__name__)

    def submit(self, request:Request, query:SiteLocationQuery):
        """
        Submits the query to the pipeline coordinator and returns the results.

        Args:
            request (Request): The request object containing the query parameters.
            query (SiteLocationQuery): The query object containing the site location and time of interest.
        """
        site_latitude = query.site_latitude
        site_longitude = query.site_longitude
        time_of_interest = query.time_of_interest
        self.log.info(f"Querying for latitude {site_latitude}, longitude {site_longitude} at time {time_of_interest}")

        # Get token from the cookie and set it in the header
        token = request.cookies.get("token")
        headers = {"Authorization": token}
        result, status_code = self.pipeline_coordinator.execute(query.model_dump(), headers=headers)

        message = result["message"]
        content = {"message": message}
        if status_code != status.HTTP_200_OK:
            return status_code, content
        
        save_results(result=result) # To local disk
        self.log.info("Results saved to disk")

        content["data"] = {"time_taken": result["time_taken"]}
        return status.HTTP_200_OK, content