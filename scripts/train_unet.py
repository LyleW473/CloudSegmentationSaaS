import set_path
import os
import torch
import mlflow
import numpy as np
import random
import matplotlib.pyplot as plt
import shutil
import json

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from multiprocessing import freeze_support

from src.ml.training.dataset import SegmentationDataset
from src.ml.training.utils import get_image_paths, calculate_dataset_statistics
from src.ml.models.unet import Model
from src.ml.training.engine import initiate_training, test_model, save_model
from src.ml.training.compose import IMAGE_TRANSFORMS

if __name__ == "__main__":
    freeze_support()

    """
    Run the following to start a local MLflow server:
    mlflow server --host 127.0.0.1 --port 8080
    """

    with mlflow.start_run():
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        
        # Training settings:
        EPOCHS = 100
        BATCH_SIZE = 32
        arch = "FPN"
        encoder_name = "timm-mobilenetv3_small_minimal_100"
        use_normalisation = True

        # Load image and mask paths
        image_paths = get_image_paths('data/archive/38-Cloud_training/train_rgbnir')
        mask_paths = get_image_paths('data/archive/38-Cloud_training/train_gt')
        print(len(image_paths), len(mask_paths))

        dataset_mean, dataset_std = calculate_dataset_statistics(image_paths)

        # Split data using 90% for training and 10% for testing
        training_image_paths, test_image_paths, training_mask_paths, test_mask_paths = train_test_split(image_paths, mask_paths, test_size=0.1, random_state=42)

        # Split the training data into training and validation sets using 80% for training and 20% for validation
        train_image_paths, val_image_paths, train_mask_paths, val_mask_paths = train_test_split(training_image_paths, training_mask_paths, test_size=0.2, random_state=42)
        print(f"Train: {len(train_image_paths)} images, {len(train_mask_paths)} masks")
        print(f"Val: {len(val_image_paths)} images, {len(val_mask_paths)} masks")
        print(f"Test: {len(test_image_paths)} images, {len(test_mask_paths)} masks")

        
        train_dataset = SegmentationDataset(
                                            train_image_paths, 
                                            train_mask_paths, 
                                            transform=IMAGE_TRANSFORMS, 
                                            use_normalisation=use_normalisation,
                                            mean=dataset_mean, 
                                            std=dataset_std
                                            )
        val_dataset = SegmentationDataset(
                                        val_image_paths, 
                                        val_mask_paths, 
                                        transform=IMAGE_TRANSFORMS, 
                                        use_normalisation=use_normalisation, 
                                        mean=dataset_mean, 
                                        std=dataset_std
                                        )
        test_dataset = SegmentationDataset(
                                        test_image_paths, 
                                        test_mask_paths, 
                                        transform=IMAGE_TRANSFORMS, 
                                        use_normalisation=use_normalisation, 
                                        mean=dataset_mean, 
                                        std=dataset_std
                                        )

        assert set(train_dataset.image_paths).isdisjoint(set(val_dataset.image_paths))
        assert set(train_dataset.image_paths).isdisjoint(set(test_dataset.image_paths))
        assert set(val_dataset.image_paths).isdisjoint(set(test_dataset.image_paths))

        train_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        val_dl = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2) 
        test_dl = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        T_MAX = EPOCHS * len(train_dl)

        # Visualisations to samples
        sample = train_dataset[0]
        print(train_dataset.image_paths[0], train_dataset.mask_paths[0])
        print(sample["image"].shape, sample["mask"].shape)
        print("Mins, maxes image", sample["image"].flatten().min(), sample["image"].flatten().max())
        print("Mins, maxes mask", sample["mask"].flatten().min(), sample["mask"].flatten().max())
        plt.subplot(1, 2, 1)
        # [C, H, W] -> [H, W, C]
        plt.imshow(sample["image"].permute(1, 2, 0))
        plt.subplot(1, 2, 2)
        # [1, H, W] -> [H, W]
        plt.imshow(sample["mask"].squeeze())
        plt.show()

        sample = val_dataset[159]
        plt.subplot(1, 2, 1)
        plt.imshow(sample["image"].permute(1, 2, 0))
        plt.subplot(1, 2, 2)
        plt.imshow(sample["mask"].squeeze())
        plt.show()  
        
        # Set up the model, optimiser and scheduler
        hyperparameters = {
                        "arch": arch, 
                        "encoder_name": encoder_name,
                        "in_channels": 4, 
                        "out_classes": 1, 
                        "t_max": T_MAX
                        }

        model = Model(**hyperparameters)
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(DEVICE)
        print(DEVICE)

        optimiser = torch.optim.Adam(model.parameters(), lr=2e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=model.T_MAX, eta_min=1e-5)

        # Save the model checkpoint
        os.makedirs("saved_models", exist_ok=True)

        # Add additional hyperparameters to the checkpoint
        hyperparameters["epochs"] = EPOCHS
        hyperparameters["batch_size"] = BATCH_SIZE
        hyperparameters["use_normalisation"] = use_normalisation

        model_num = len(os.listdir("saved_models"))
        model_dir = f"saved_models/model_{model_num}"
        os.makedirs(model_dir, exist_ok=True)

        # Train the model
        best_model_dict, metrics = initiate_training(
                                    model=model, 
                                    optimiser=optimiser, 
                                    scheduler=scheduler, 
                                    train_dl=train_dl, 
                                    val_dl=val_dl, 
                                    hyperparameters=hyperparameters, 
                                    device=DEVICE,
                                    model_dir=model_dir
                                    )
        best_model = best_model_dict["model"]
        best_model_metadata = best_model_dict["metadata"]
        train_metrics = metrics["training"]
        validation_metrics = metrics["validation"]
        print(f"Training metrics: {train_metrics}")
        print(f"Validation metrics: {validation_metrics}")
        
        # Test the model
        test_metrics = test_model(model=best_model, test_dl=test_dl, device=DEVICE)
        print(f"Test metrics: {test_metrics}")

        # Set up MLflow for logging
        mlflow.set_tracking_uri("http://127.0.0.1:8080")
        mlflow.set_experiment("unet-cloud-segmentation")
        mlflow.set_tag("Training Info", "Training a U-Net model for cloud segmentation")
        mlflow.log_params(hyperparameters)
        mlflow.log_artifact(model_dir)

        # Automating uploading the best model to the 'model_weights' directory, which the core system when deployed
        shutil.rmtree("model_weights", ignore_errors=True)
        os.makedirs("model_weights", exist_ok=True)
        save_model(model=best_model, hyperparameters=hyperparameters, save_path=f"model_weights/best_model.pt")
        with open("model_weights/best_model_metadata.json", "w") as f:
            json.dump(best_model_metadata, f)