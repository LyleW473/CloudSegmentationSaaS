import set_path
import numpy as np

from src.ml.training.satellite_image_processor import SatelliteImageProcessor
from PIL import Image

if __name__ == "__main__":
    
    # Training images
    base_dir = "data/archive/38-Cloud_training"
    patch_csv_file_name = "training_patches_38-Cloud.csv"
    image_processor = SatelliteImageProcessor()
    image_processor.create_rgbnir_images(
                                        base_dir=base_dir,
                                        patch_csv_file_name=patch_csv_file_name,
                                        is_train=True
                                        )
    test_image = Image.open("data/archive/38-Cloud_training/train_rgbnir/rgbnir_patch_8_1_by_8_LC08_L1TP_061017_20160720_20170223_01_T1.tif")
    test_image.show()
    print(np.array(test_image).shape)

    # Test images
    base_dir = "data/archive/38-Cloud_test"
    patch_csv_file_name = "test_patches_38-Cloud.csv"
    image_processor.create_rgbnir_images(
                                        base_dir=base_dir,
                                        patch_csv_file_name=patch_csv_file_name,
                                        is_train=False
                                        )