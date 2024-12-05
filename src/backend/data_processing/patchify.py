import numpy as np

from typing import Tuple

class Patchify:
    def __call__(self, image: np.ndarray, patch_size:Tuple[int, int]) -> np.ndarray:
        """
        Splits an image into patches of a specified size.
        - Returns the patches in the format (num_patches, patch_height, patch_width, channels).
        
        Args:
            image (np.ndarray): The image to split into patches.
            patch_size (int): The size of the patches to create.

        Returns:
            List[np.ndarray]: A list of patches.
        """
        patch_height, patch_width = patch_size
        patches = []
        for i in range(0, image.shape[0], patch_height):
            for j in range(0, image.shape[1], patch_width):
                patch = image[i:i+patch_height, j:j+patch_width]

                # Pad the patch if it is not the correct size
                if patch.shape[0] != patch_height or patch.shape[1] != patch_width:
                    pad_height = patch_height - patch.shape[0]
                    pad_width = patch_width - patch.shape[1]
                    patch = np.pad(patch, ((0, pad_height), (0, pad_width), (0, 0)), mode="constant", constant_values=0)
                patches.append(patch)

        return np.stack(patches, axis=0)