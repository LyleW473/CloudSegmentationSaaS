import set_path
import os
import torch
import mlflow
import numpy as np
import random
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from multiprocessing import freeze_support

from src.ml.training.dataset import SegmentationDataset
from src.ml.training.utils import get_image_paths
from src.ml.training.engine import test_model, load_trained_model
from src.ml.training.compose import IMAGE_TRANSFORMS

if __name__ == "__main__":
    freeze_support()

    with mlflow.start_run():
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

        # Load trained model
        trained_model_num = 0
        trained_model_epoch = 5
        model_dir = f"saved_models/model_{trained_model_num}"
        model, checkpoint = load_trained_model(model_dir=model_dir, epoch_num=trained_model_epoch)
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(DEVICE)
        print(DEVICE)
        
        hyperparameters = checkpoint["hyperparams"]
        BATCH_SIZE = checkpoint["hyperparams"]["batch_size"]
        
        # Load image and mask paths
        image_paths = get_image_paths('data/archive/38-Cloud_training/train_rgbnir')
        mask_paths = get_image_paths('data/archive/38-Cloud_training/train_gt_processed')
        print(len(image_paths), len(mask_paths))

        # Split data using 90% for training and 10% for testing
        training_image_paths, test_image_paths, training_mask_paths, test_mask_paths = train_test_split(image_paths, mask_paths, test_size=0.1, random_state=42)

        # Split the training data into training and validation sets using 80% for training and 20% for validation
        train_image_paths, val_image_paths, train_mask_paths, val_mask_paths = train_test_split(training_image_paths, training_mask_paths, test_size=0.2, random_state=42)
        print(f"Train: {len(train_image_paths)} images, {len(train_mask_paths)} masks")
        print(f"Val: {len(val_image_paths)} images, {len(val_mask_paths)} masks")
        print(f"Test: {len(test_image_paths)} images, {len(test_mask_paths)} masks")

        test_dataset = SegmentationDataset(test_image_paths, test_mask_paths, transform=IMAGE_TRANSFORMS)
        test_dl = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        
        # Test the model
        test_metrics = test_model(model=model, test_dl=test_dl, device=DEVICE)
        print(f"Test metrics: {test_metrics}")

        # Set up MLflow for logging
        mlflow.set_tracking_uri("http://127.0.0.1:8080")
        mlflow.set_experiment("unet-cloud-segmentation")
        mlflow.set_tag("Testing Info", f"Model {trained_model_num} at epoch {trained_model_epoch}")
        mlflow.log_params(hyperparameters)
        mlflow.log_artifact(model_dir)

    # Visualise the model predictions
    os.makedirs("sample_predictions", exist_ok=True)
    num_added = 0
    num_test_samples = 100
    for i, batch in enumerate(test_dl):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        images = batch["image"]
        masks = batch["mask"]
        original_images = batch["original_image"]

        # Get model predictions
        with torch.no_grad():
            preds = model(images)
            # Convert to probabilities for each pixel
            preds = torch.sigmoid(preds)
            print(preds.shape, preds[0].flatten().min(), preds[0].flatten().max())

            converted_preds = preds.detach().cpu().numpy() * 255
            print(converted_preds.shape, converted_preds[0].flatten().min(), converted_preds[0].flatten().max())

        flag = False
        for j in range(len(images)):
            plt.figure(figsize=(10, 10))
            plt.subplot(1, 3, 1)
            plt.imshow(original_images[j].cpu().permute(1, 2, 0))
            plt.title("Image")

            plt.subplot(1, 3, 2)
            plt.imshow(masks[j].cpu().squeeze())
            plt.title("Mask")

            plt.subplot(1, 3, 3)
            plt.imshow(converted_preds[j].squeeze())
            plt.title("Predicted Mask")

            plt.savefig(f"sample_predictions/{num_added}.png")

            num_added += 1
            if num_added == num_test_samples:
                flag = True
                break
        if flag:
            break