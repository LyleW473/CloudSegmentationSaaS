import torch
import segmentation_models_pytorch as smp
import mlflow
import json

from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple, List, Union

from src.ml.models.unet import Model

def initiate_training(
                    model:torch.nn.Module,
                    optimiser:torch.optim.Optimizer,
                    scheduler:torch.optim.lr_scheduler._LRScheduler,
                    train_dl:DataLoader,
                    val_dl:DataLoader,
                    hyperparameters:Dict[str, Any],
                    device:torch.device,
                    model_dir:str
                    ) -> Tuple[torch.nn.Module, Dict[str, Dict[str, Any]]]:
    """
    Core function to train and validate a model.

    Args:
        model (torch.nn.Module): The model to train and validate.
        optimiser (torch.optim.Optimizer): The optimiser to use.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The scheduler to use.
        train_dl (DataLoader): The data loader to use for training.
        val_dl (DataLoader): The data loader to use for validation.
        hyperparameters (Dict[str, Any]): The hyperparameters used for training and the model set-up.
        device (torch.device): The device to train and validate the model on.
        model_dir (str): The directory to save all of the model checkpoints in.
    """
    
    training_metrics = {"loss": [], "iou": []}
    validation_metrics = {"loss": [], "iou": []}

    for epoch in range(hyperparameters['epochs']):

        # Training loop
        model.train()
        train_description = f"Training | Epoch {epoch+1}/{hyperparameters['epochs']}"
        train_loss, train_tp, train_fp, train_fn, train_tn = forward_epoch_pass(model, optimiser, train_dl, device, "train", train_description)
        
        # Update scheduler
        scheduler.step()

        # Calculate training metrics
        train_loss /= len(train_dl)
        train_iou = smp.metrics.iou_score(torch.tensor([train_tp]), torch.tensor([train_fp]), torch.tensor([train_fn]), torch.tensor([train_tn]), reduction="micro")
        training_metrics["loss"].append(train_loss)
        training_metrics["iou"].append(train_iou)

        print(f"Epoch {epoch+1}/{hyperparameters['epochs']}, Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}")

        # Validation loop
        model.eval()
        with torch.no_grad():
            val_description = f"Validation | Epoch {epoch+1}/{hyperparameters['epochs']}"
            val_loss, val_tp, val_fp, val_fn, val_tn = forward_epoch_pass(model, None, val_dl, device, "valid", val_description)

        # Calculate validation metrics
        val_loss /= len(val_dl)
        val_iou = smp.metrics.iou_score(torch.tensor([val_tp]), torch.tensor([val_fp]), torch.tensor([val_fn]), torch.tensor([val_tn]), reduction="micro")
        validation_metrics["loss"].append(val_loss)
        validation_metrics["iou"].append(val_iou)

        print(f"Epoch {epoch+1}/{hyperparameters['epochs']}, Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")

        # Logging metrics to MLFlow
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("train_iou", train_iou, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_iou", val_iou, step=epoch)

        # Save the model checkpoint at the end of each epoch
        save_model(model=model, hyperparameters=hyperparameters, save_path=f"{model_dir}/epoch_{epoch+1}.pt")

        # Select the best model based on the validation loss and IoU (After each epoch, just to get a sense of how the model is performing)
        best_model_dict = select_best_model(
                                            model_dir=model_dir,
                                            losses=validation_metrics["loss"],
                                            ious=validation_metrics["iou"]
                                            )
        print(f"Best model found at epoch {best_model_dict['metadata']['best_epoch']}, with loss {best_model_dict['metadata']['best_loss']:.4f} and IoU {best_model_dict['metadata']['best_iou']:.4f}")
        if epoch != (hyperparameters['epochs'] - 1):
            del best_model_dict["model"] # Delete the model to save memory

    return best_model_dict, {"training": training_metrics, "validation": validation_metrics}

def save_model(model:torch.nn.Module, hyperparameters:Dict[str, Any], save_path:str):
    """
    Function to save a model to a specified path.

    Args:
        model (torch.nn.Module): The model to save.
        hyperparameters (Dict[str, Any]): The hyperparameters used for training and the model set-up.
        save_path (str): The path to save the model to.
    """
    checkpoint = {"model": model.state_dict(), "hyperparams": hyperparameters}
    torch.save(checkpoint, save_path)

def select_best_model(model_dir, losses:List[float], ious:List[float]) -> Tuple[int, float, float]:
    """
    Function to select the best model based on the validation loss and IoU.

    Args:
        model_dir (str): The directory where the model checkpoints are stored.
        losses (List[float]): A list of validation losses for each epoch.
        ious (List[float]): A list of validation IoU scores for each epoch.
    """
    losses_tensor = torch.tensor(losses)
    ious_tensor = torch.tensor(ious)

    # Determine the best epoch based on the combined score of normalised losses and inverted IoU scores
    best_loss = losses_tensor.min()
    best_iou = ious_tensor.max()

    normalised_losses = (losses_tensor - best_loss) / (losses_tensor.max() - best_loss)
    normalised_ious = (ious_tensor - ious_tensor.min()) / (best_iou - ious_tensor.min())
    inverted_ious = 1 - normalised_ious
    combined_scores = normalised_losses + inverted_ious
    best_epoch = combined_scores.argmin().item()
    
    best_model, best_checkpoint = load_trained_model(model_dir=model_dir, epoch_num=best_epoch+1)
    best_loss = best_loss.item()
    best_iou = best_iou.item()

    # Save metadata in the model directory
    metadata = {
                "best_epoch": best_epoch+1,
                "best_loss": best_loss,
                "best_iou": best_iou,
                "hyperparameters": best_checkpoint["hyperparams"]
                }
    with open(f"{model_dir}/metadata.json", "w") as f:
        json.dump(metadata, f)
        
    result = {
            "model": best_model,
            "metadata": metadata
            }
    return result

def load_trained_model(model_dir:str, epoch_num:int) -> Union[Tuple[torch.nn.Module, Dict[str, Any]], None]:
    """
    Function to load a trained model from a checkpoint.

    Args:
        model_dir (str): The directory where the model checkpoints are stored.
        epoch_num (int): The epoch number to load the model from.
    """
    try:
        checkpoint = torch.load(f"{model_dir}/epoch_{epoch_num}.pt")
        model = Model(
                    arch=checkpoint["hyperparams"]["arch"], 
                    encoder_name=checkpoint["hyperparams"]["encoder_name"], 
                    in_channels=4, 
                    out_classes=1, 
                    t_max=checkpoint["hyperparams"]["t_max"]
                    )
        model.load_state_dict(checkpoint["model"])
        return model, checkpoint
    except FileNotFoundError:
        print(f"Model checkpoint not found at {model_dir}/epoch_{epoch_num}.pt")
        return None

def forward_epoch_pass(
                    model:torch.nn.Module,
                    optimiser:torch.optim.Optimizer,
                    dl:DataLoader,
                    device:torch.device,
                    mode:str,
                    description:str,
                    ) -> Tuple[float, float, float, float]:
    """
    Helper function to calculate the loss and metrics for a single epoch.
    Calculates:
    - Loss
    - True Positives (TP)
    - False Positives (FP)
    - False Negatives (FN)
    - True Negatives (TN)

    Args:
        model (torch.nn.Module): The model to train and validate.
        dl (DataLoader): The data loader to use for training or validation.
        device (torch.device): The device to train and validate the model on.
        mode (str): The mode to run the model in, either "train" or "valid".
        description (str): The description for the tqdm progress bar.
    """
    total_loss = 0.0
    tp, fp, fn, tn = 0, 0, 0, 0

    for batch in tqdm(dl, desc=description):
        if mode == "train":
            optimiser.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model.shared_step(batch, mode)

        if mode == "train":
            loss = outputs["loss"]
            loss.backward()
            optimiser.step()

        total_loss += outputs["loss"].item()
        tp += outputs["tp"].sum().item()
        fp += outputs["fp"].sum().item()
        fn += outputs["fn"].sum().item()
        tn += outputs["tn"].sum().item()
    return total_loss, tp, fp, fn, tn

def test_model(
            model:torch.nn.Module,
            test_dl:DataLoader,
            device:torch.device
            ) -> Dict[str, float]:
    """
    Function to test the model on some unseen test data.

    Args:
        model (torch.nn.Module): The model to test.
        test_dl (DataLoader): The data loader to use for testing.
        device (torch.device): The device to test the model on.
    """
    model = model.to(device)
    model.eval()
    test_description = "Testing"
    test_loss, test_tp, test_fp, test_fn, test_tn = forward_epoch_pass(model, None, test_dl, device, "test", test_description)

    # Calculate test metrics
    test_loss /= len(test_dl)
    test_iou = smp.metrics.iou_score(torch.tensor([test_tp]), torch.tensor([test_fp]), torch.tensor([test_fn]), torch.tensor([test_tn]), reduction="micro")
    
    # Logging final test metrics to MLFlow
    mlflow.log_metric("test_loss", test_loss)
    mlflow.log_metric("test_iou", test_iou)
    
    print(f"Test Loss: {test_loss:.4f}, Test IoU: {test_iou:.4f}")
    return {"loss": test_loss, "iou": test_iou}