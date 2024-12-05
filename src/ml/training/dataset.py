import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Callable
from torchvision.transforms import v2

class SegmentationDataset(torch.utils.data.Dataset):
    """
    Dataset class for cloud segmentation.
    """
    def __init__(
                self, 
                image_paths:List[str], 
                mask_paths:List[str], 
                transform:Optional[Callable]=None, 
                use_normalisation:Optional[bool]=False,
                mean:Optional[List[float]]=None,
                std:Optional[List[float]]=None
                ):
        """
        Initialises the dataset object.

        Args:
            image_paths (List[str]): A list containing the paths to each satellite image.
            mask_paths (List[str]): A list containing the paths to each corresponding mask image.
            transform (Optional[Callable], optional): Image transformation function. Defaults to None.
            use_normalisation (Optional[bool], optional): Whether to normalise the image data. Defaults to False.
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

        # Normalised with ImageNet statistics (with an additional channel for NIR, taken from red channel)
        if mean is None and std is None:
            mean = [0.485, 0.456, 0.406, 0.485]
            std = [0.229, 0.224, 0.225, 0.229]
        assert not(mean is None and std is not None), "Both mean and std must be provided if one is provided."
        assert not(mean is not None and std is None), "Both mean and std must be provided if one is provided."
        self.normalisation_transform = v2.Normalize(mean=mean, std=std)
        self.use_normalisation = use_normalisation

    def __len__(self):
        return len(self.image_paths)
    
    def _prepare_mask(self, mask_pil:Image) -> np.ndarray:
        """
        Prepares the mask prior to applying the transform.

        Args:
            mask_pil (PIL.Image): The mask image as a PIL image.
        """
        mask = np.array(mask_pil)
        mask = np.expand_dims(mask, axis=-1) # (H, W) -> (H, W, 1)
        return mask

    def __getitem__(self, idx:int) -> Dict[str, torch.Tensor]:
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = Image.open(image_path)
        original_image = np.array(image)
        mask = Image.open(mask_path)

        if self.transform:
            image = np.array(image)
            mask = self._prepare_mask(mask_pil=mask)

            combined_image = np.concatenate((image, mask), axis=-1) # [H, W, C]
            combined_image = self.transform(combined_image) # [C, H, W]

            image = combined_image[:4, :, :] # (R, G, B, NIR)
            mask = combined_image[-1, :, :].unsqueeze(0) # (1, H, W)

        if not isinstance(image, torch.Tensor):
            image = torch.tensor(np.array(image)).permute(2, 0, 1)
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(np.array(mask)).unsqueeze(0) # Add channel dimension (1, H, W)
        
        original_image = torch.tensor(original_image / 255.0).permute(2, 0, 1)

        if self.use_normalisation:
            image = self.normalisation_transform(image)

        return {
            "image": image.to(dtype=torch.float32), 
            "mask": mask.to(dtype=torch.float32), 
            "original_image": original_image.to(dtype=torch.float32)
            }