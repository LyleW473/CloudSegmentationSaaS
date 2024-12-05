import numpy as np

from PIL import Image

def visualise_image(image:np.ndarray):
    """
    Given an image, visualises the image using the PIL library.
    
    Args:
        image (np.ndarray): The image to visualise.
    """
    image = Image.fromarray(image)
    image.show()

def visualise_image_with_mask(image:np.ndarray, mask:np.ndarray):
    """
    Given an image and a mask, visualises the image and mask side-by-side.
    
    Args:
        image (np.ndarray): The image to visualise.
        mask (np.ndarray): The mask to visualise.
    """
    image = Image.fromarray(image)
    mask = Image.fromarray(mask)

    if mask.ndim == 2:
        mask = np.stack([mask, mask, mask], axis=-1) # (H, W) -> (H, W, 3)

    combined = np.concatenate([image, mask], axis=1)
    combined = Image.fromarray(combined)
    combined.show()