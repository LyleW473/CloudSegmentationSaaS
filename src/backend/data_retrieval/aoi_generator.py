import numpy as np

from typing import List, Tuple

class AOIGenerator:
    """
    Class for generating a bounding box around a site of interest in
    the format [min_lon, min_lat, max_lon, max_lat].
    """
    def __init__(self):
        self.GROUND_SAMPLE_DISTANCE = 30.0
        self.APPROXIMATE_METERS_PER_DEGREE = 111111.0

    def __call__(self, site_latitude:float, site_longitude:float, desired_img_size:Tuple[int, int]) -> List[float]:
        """
        Given a site latitude and longitude, returns a bounding box around the site.
        - Assumes that the site is at the center of the image.
        
        Args:
            site_latitude (float): The latitude of the site (degrees).
            site_longitude (float): The longitude of the site (degrees).
            desired_img_size (Tuple[int, int]): The desired size of the image in the format (height, width).
        """
        # Calculate the buffer size in meters
        desired_img_height, desired_img_width = desired_img_size
        half_buffer_height = (desired_img_height / 2) * self.GROUND_SAMPLE_DISTANCE
        half_buffer_width = (desired_img_width / 2) * self.GROUND_SAMPLE_DISTANCE

        # Calculate the buffer size in degrees
        lat_buffer_degrees = half_buffer_height / self.APPROXIMATE_METERS_PER_DEGREE
        lon_buffer_degrees = half_buffer_width / (self.APPROXIMATE_METERS_PER_DEGREE * np.cos(np.radians(site_latitude)))

        # Calculate the bounding box around the site.
        min_lon = site_longitude - lon_buffer_degrees
        max_lon = site_longitude + lon_buffer_degrees
        min_lat = site_latitude - lat_buffer_degrees
        max_lat = site_latitude + lat_buffer_degrees
        return [min_lon, min_lat, max_lon, max_lat]