import pystac_client
import planetary_computer
import odc.stac
import matplotlib.pyplot as plt
import numpy as np

from xarray.core.dataset import Dataset
from pystac.item_collection import ItemCollection
from pystac.item import Item
from pystac.extensions.eo import EOExtension as eo
from typing import List, Dict
from src.backend.data_retrieval.aoi_generator import AOIGenerator

class DataRetriever:
    """
    Class for retrieving Landsat-8 data from the Planetary Computer STAC API.
    """
    def __init__(self):
        self.catalog = pystac_client.Client.open(
                                                "https://planetarycomputer.microsoft.com/api/stac/v1",
                                                modifier=planetary_computer.sign_inplace,
                                                )
        self.bands_of_interest = ["nir08", "red", "green", "blue"]
        self.aoi_generator = AOIGenerator()
        self.IMAGE_SIZE = (768, 768)
        self.MAX_ITEMS = 1
        self.MAX_CLOUD_COVER = 10

    def _sort_by_most_cloudy(self, items:ItemCollection) -> ItemCollection:
        """
        Sorts the collection of items by cloud cover in descending order
        (most cloudy first).
        
        Args:
            items (ItemCollection): The collection of items to choose from.
        """
        sorted_items = sorted(items, key=lambda item: eo.ext(item).cloud_cover, reverse=True)
        return sorted_items
    
    def _extract_bands(self, data:Dataset, bands_of_interest:List[str]) -> Dict[str, np.ndarray]:
        """
        Extracts the bands from an xarray dataset and returns a dictionary
        mapping the band names to numpy arrays of the band data.

        Args:
            data (Dataset): The xarray dataset containing the bands of interest.
            bands_of_interest (List[str]): The list of bands to extract from the dataset.
        """
        data_dict = {}
        for band in bands_of_interest:
            data_array = data[band].values.squeeze(0) # (1, H, W) -> (H, W)
            data_dict[band] = data_array.tolist() # For JSON serialisation
        return data_dict
    
    def _search_for_items(self, bbox_of_interest:List[float], time_of_interest:str) -> ItemCollection:
        """
        Given a bounding box and time range of interest, searches the catalog for Landsat-8 images.

        Args:
            bbox_of_interest (List[float]): The bounding box of interest in the format [min_lon, min_lat, max_lon, max_lat].
            time_of_interest (str): The time range of interest in the format "YYYY-MM-DD/YYYY-MM-DD".
        
        """
        search = self.catalog.search(
                                    collections=["landsat-c2-l2"],
                                    bbox=bbox_of_interest,
                                    datetime=time_of_interest,
                                    query={ "eo:cloud_cover": {"lt": self.MAX_CLOUD_COVER},
                                            "platform": {"in": ["landsat-8"]},
                                            },
                                    )
        items = search.get_all_items()
        return items

    def retrieve_data(self, site_latitude:float, site_longitude:float, time_of_interest:str) -> List[Dict[str, np.ndarray]]:
        """
        Given a site of interest (lat, lon), retrieves the least cloudy Landsat-8 image and extracts the bands of 
        interest from the image.

        Args:
            site_latitude (float): The (centre) latitude of the site of interest.
            site_longitude (float): The (centre) longitude of the site of interest.
            time_of_interest (str): The time range of interest in the format "YYYY-MM-DD/YYYY-MM-DD".
        """
        # Generate the bounding box around the site
        bbox_of_interest = self.aoi_generator(
                                            site_latitude=site_latitude,
                                            site_longitude=site_longitude,
                                            desired_img_size=self.IMAGE_SIZE
                                            )

        # Search for items in the catalog
        items = self._search_for_items(
                                        bbox_of_interest=bbox_of_interest, 
                                        time_of_interest=time_of_interest
                                        )
        if len(items) == 0:
            extracted_bands_dicts = None
        else:
            sorted_items = self._sort_by_most_cloudy(items)
            sorted_items = sorted_items[:self.MAX_ITEMS]

            extracted_bands_dicts = []
            metadatas = []
            for i in range(0, len(sorted_items)):
                # Load the data for the selected item
                data = odc.stac.load(
                                    items=[sorted_items[i]],
                                    bands=self.bands_of_interest, 
                                    bbox=bbox_of_interest
                                    )
                
                # Extract the bands from the dataset
                extracted_bands_dict = self._extract_bands(
                                                        data=data,
                                                        bands_of_interest=self.bands_of_interest
                                                        )

                metadata = {
                            "date": str(sorted_items[i].datetime.strftime("%Y-%m-%d")), # In the format "YYYY-MM-DD"
                            "cloud_coverage": eo.ext(sorted_items[i]).cloud_cover
                            }
                metadatas.append(metadata)
                extracted_bands_dicts.append(extracted_bands_dict)
        return extracted_bands_dicts, metadatas
    
if __name__ == "__main__":
    data_retriever = DataRetriever()
    bbox_of_interest = [-122.2751, 47.5469, -121.9613, 47.7458]
    time_of_interest = "2021-01-01/2021-12-31"
    data = data_retriever.retrieve_data(
                                        bbox_of_interest=bbox_of_interest,
                                        time_of_interest=time_of_interest
                                        )
    for band, band_data in data.items():
        plt.imshow(band_data)
        plt.title(band)
        plt.show()