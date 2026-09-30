import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

from callbacks.early_stopping import EarlyStopping


def _keep_frozen_batchnorm_in_eval(model):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            if not any(parameter.requires_grad for parameter in module.parameters()):
                module.eval()

def train_one_epoch(
        model, 
        dataloader,
        criterion,
        optimizer,
        device,
        ):
    
    model.train()
    _keep_frozen_batchnorm_in_eval(model)

    total_loss = 0.0
    total_samples = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels) # loss function - funcion de perdida

        optimizer.zero_grad() # limpiar gradientes
        loss.backward() # backpropagation
        optimizer.step() # actualizar pesos

        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

    avg_loss = total_loss / total_samples
    return avg_loss  

def evaluate(
        model, 
        dataloader,
        criterion,
        device,
        ):
    
    model.eval() # modo evaluacion

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad(): # desactiva el calculo de gradientes
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels) # loss function - funcion de perdida

            total_loss += loss.item() * inputs.size(0)
            total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    return avg_loss


def evaluate_with_report(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            predictions = outputs.argmax(dim=1)

            total_loss += loss.item() * inputs.size(0)
            total_samples += labels.size(0)
            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    average_loss = total_loss / total_samples
    return {
        "loss": average_loss,
        "accuracy": accuracy_score(all_labels, all_predictions),
        "precision": precision_score(all_labels, all_predictions, average=None, labels=[0, 1], zero_division=0).tolist(),
        "recall": recall_score(all_labels, all_predictions, average=None, labels=[0, 1], zero_division=0).tolist(),
        "f1": f1_score(all_labels, all_predictions, average=None, labels=[0, 1], zero_division=0).tolist(),
        "f1_macro": f1_score(all_labels, all_predictions, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(all_labels, all_predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            all_labels,
            all_predictions,
            output_dict=True,
            zero_division=0,
        ),
    }


def save_metrics_report(metrics, model_name, output_dir="artifacts"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report = {"model_name": model_name, **metrics}
    report_path = output_path / f"metrics_{model_name}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path

def fit(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs,
    early_stopping= None,
    scheduler=None,
    checkpoint_path=None,
    phase_name="phase",
):
    history = {
        "train_loss": [],
        "val_loss": [],
        "learning_rate": [],
        "val_f1_macro": [],
        "val_recall_defective": [],
    }

    
    for epoch in range(epochs):
        current_lr = optimizer.param_groups[0]['lr']
        train_loss = train_one_epoch(
            model, 
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_metrics = evaluate_with_report(model, val_loader, criterion, device)
        val_loss = val_metrics["loss"]
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["learning_rate"].append(current_lr)
        history["val_f1_macro"].append(val_metrics["f1_macro"])
        history["val_recall_defective"].append(val_metrics["recall"][1])

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_metrics['f1_macro']:.4f}, Val Recall defective: {val_metrics['recall'][1]:.4f}, Learning Rate: {current_lr:.6e}")

        if scheduler is not None:
            scheduler.step(val_loss)  # Update learning rate based on validation loss

        if early_stopping is not None:
            early_stopping(model, val_metrics["f1_macro"])
            if checkpoint_path is not None and early_stopping.best_model_state is not None:
                torch.save({
                    "model_state_dict": early_stopping.best_model_state,
                    "best_val_f1_macro": early_stopping.best_score,
                    "best_val_loss": val_loss,
                    "best_val_recall_defective": val_metrics["recall"][1],
                    "phase": phase_name,
                    "best_epoch": len(history["val_f1_macro"]),
                }, checkpoint_path)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break
    if early_stopping is not None:
        early_stopping.restore_best_model(model)
        print("Restored best model state.")

    return history, early_stopping.best_score if early_stopping is not None else None