import torch
from torchvision.transforms import v2

# Define image/mask transformations
IMAGE_TRANSFORMS = v2.Compose([
                                v2.ToImage(), # Convert to PIL image
                                
                                # Data augmentation:
                                v2.RandomHorizontalFlip(p=0.5),
                                v2.RandomVerticalFlip(p=0.5),

                                v2.ToDtype(torch.float32, scale=True),
                                ])