import torch
import numpy as np
import json

from torchvision.transforms import v2
from typing import Tuple

from src.ml.models.unet import Model
from src.backend.utils.exceptions import ModelLoadingException, ModelInferenceException

def _load_model(checkpoint_path:str="model_weights/best_model.pt") -> torch.nn.Module:
    """
    Loads a pre-trained model from a checkpoint and returns it.

    Args:
        checkpoint_path (str): The path to the model checkpoint. Default is "model_weights/best_model.pt".
    """
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu")) # Load on CPU for inference
    weights = checkpoint["model"] # Model state dict
    hyperparams = checkpoint["hyperparams"] # Model hyperparameters
    model = Model(
                arch=hyperparams["arch"], 
                encoder_name=hyperparams["encoder_name"],
                in_channels=hyperparams["in_channels"], 
                out_classes=hyperparams["out_classes"],
                t_max=hyperparams["t_max"]
                )
    model.load_state_dict(weights)
    return model, hyperparams

def _add_padding(data:np.ndarray, padding:int) -> np.ndarray:
    """
    Adds padding to the data along the first axis.
    """
    return np.pad(data, ((0, padding), (0, 0), (0, 0), (0, 0)), mode="constant")

def _convert_to_batches(data:np.ndarray, batch_size:int) -> Tuple[np.ndarray, int]:
    """
    Converts the input data into batches of the specified batch size.
    - Returns the data in the format (num_batches, batch_size, height, width, channels).

    Args:
        data (np.ndarray): The input data to be batched.
        batch_size (int): The desired batch size.
    """
    num_patches = data.shape[0]

    # Add padding when necessary (affects model performance due to batch normalization)
    if num_patches < batch_size:
        n_padding = batch_size - num_patches
        data = _add_padding(data=data, padding=n_padding)
    else:
        remainder = num_patches % batch_size
        if remainder > 0:
            n_padding = batch_size - remainder
            data = _add_padding(data=data, padding=n_padding)
    
    # Reshape data into the form: (num_batches, batch_size, height, width, channels)
    batched_data = np.reshape(data, (-1, batch_size, data.shape[1], data.shape[2], data.shape[3]))
    return batched_data, n_padding

def get_model_predictions(data:np.ndarray) -> np.ndarray:
    """
    Gets the predicted segmentation masks from the model for the input data.
    - Returns a single numpy array of the shape (num_patches, 1, height, width).

    Args:
        data (np.ndarray): The input data to perform inference on.
    """
    # Load model and training hyperparameters
    try:
        model, hyperparams = _load_model()
        model.eval()
    except Exception as e:
        raise ModelLoadingException(f"Error loading the model, please check the model path. {e}")
    
    try:
        # Convert into batches for inference
        batched_data, n_padding = _convert_to_batches(data, batch_size=hyperparams["batch_size"])

        if hyperparams["use_normalisation"]:
            with open(f"dataset_statistics/stats.json", "r") as f:
                stats = json.load(f)
                mean, std = stats["mean"], stats["std"]
                image_transform= v2.Normalize(mean=mean, std=std)

        # Perform inference on the data
        all_preds = []
        for batch in batched_data:
            batch = torch.tensor(batch).float()
            batch = batch.permute(0, 3, 1, 2) # [B, H, W, C] -> [B, C, H, W]

            if hyperparams["use_normalisation"]:
                batch = torch.stack([image_transform(image) for image in batch], dim=0)

            with torch.no_grad():
                preds = model(batch)
                # Convert to probabilities for each pixel
                preds = torch.sigmoid(preds)
                converted_preds = (preds * 255) # Convert to [0, 255] for visualisation
            all_preds.append(converted_preds)
        
        # Concatenate the predictions
        all_preds = torch.cat(all_preds, dim=0)

        # Remove all padding predictions
        all_preds = all_preds[:-n_padding]

        # Convert to numpy array and remove channel dimension
        all_preds = all_preds.detach().cpu().numpy()
        all_preds = np.squeeze(all_preds, axis=1) # [B, 1, H, W] -> [B, H, W]
        return all_preds
    except Exception as e:
        raise ModelInferenceException(f"Error during model inference, please check the input data. {e}")