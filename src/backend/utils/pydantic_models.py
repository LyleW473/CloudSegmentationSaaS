from pydantic import BaseModel, Field
from typing import Union, Annotated, Dict, List

# Type aliases
FloatOrInt = Union[float, int]
ImageArray2D = List[List[FloatOrInt]] # (height, width)
ImageArray3D = List[ImageArray2D] # (num_patches, height, width)
ImageArray4D = List[ImageArray3D] # (num_patches, height, width, num_channels)

# Pydantic models for each service
class SiteLocationQuery(BaseModel):
    site_latitude: Annotated[FloatOrInt, Field(ge=-90, le=90)]
    site_longitude: Annotated[FloatOrInt, Field(ge=-180, le=180)]
    time_of_interest: Annotated[str, Field(pattern=r'^\d{4}-\d{2}-\d{2}/\d{4}-\d{2}-\d{2}$')] # Format: YYYY-MM-DD/YYYY-MM-DD

class ExtractedBandsQuery(BaseModel):
    data: Annotated[Dict[str, ImageArray2D], # A dictionary mapping band names to 2-D image arrays of shape (height, width) represented as a list of lists.
                    Field(description="The dictionary of extracted bands.")] 
    
class PatchImagesQuery(BaseModel):
    data: Annotated[ImageArray4D, # A 4-D image array of shape (num_patches, height, width, num_channels) represented as a list of lists.
                    Field("The patches extracted from the combined image represented as a 4-D array.")]
    
class PatchSegmentationMasksQuery(BaseModel):
    data: Annotated[ImageArray3D, # A 3-D image array of shape (num_patches, height, width) represented as a list of lists.
                    Field(description="The segmentation masks for each of the extracted patches.")]
    stride: Annotated[FloatOrInt, Field(description="The stride used to extract the patches.")]
    patch_size: Annotated[FloatOrInt, Field(description="The size of the patches extracted.")]
    original_img_height: Annotated[FloatOrInt, Field(description="The height of the original image.")]
    original_img_width: Annotated[FloatOrInt, Field(description="The width of the original image.")]