import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from callbacks.early_stopping import EarlyStopping
from datasets.main import SPLIT_MANIFEST_PATH, get_elpv_loaders
from engine.trainer import evaluate_with_report, fit, save_metrics_report
from models.factory import create_model
from utils.device import get_device


CONFIG = {
    "batch_size": 32,
    "weight_decay": 1e-4,
    "phase1_epochs": 4,
    "phase2_epochs": 20,
    "phase1_lr": 1e-3,
    "phase2_head_lr": 1e-4,
    "phase2_block_lr": 1e-5,
    "patience": 5,
    "min_delta": 1e-3,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parameter_groups(model, phase, config):
    head_module = model.network.fc if model.name == "resnet18" else model.network.classifier[-1]
    head_ids = {id(parameter) for parameter in head_module.parameters()}
    head_parameters = [parameter for parameter in model.parameters()
                       if parameter.requires_grad and id(parameter) in head_ids]
    block_parameters = [parameter for parameter in model.parameters()
                        if parameter.requires_grad and id(parameter) not in head_ids]
    groups = [{"params": head_parameters, "lr": config["phase1_lr"]}]
    if phase == "finetune":
        groups[0]["lr"] = config["phase2_head_lr"]
        groups.append({"params": block_parameters, "lr": config["phase2_block_lr"]})
    return groups


def verify_model_step(model, loader, criterion, device):
    model.train()
    inputs, labels = next(iter(loader))
    outputs = model(inputs.to(device))
    if tuple(outputs.shape) != (inputs.shape[0], 2):
        raise ValueError(f"Expected model output [{inputs.shape[0]}, 2], got {tuple(outputs.shape)}")
    criterion(outputs, labels.to(device)).backward()
    model.zero_grad(set_to_none=True)
    print(f"Verification: output={tuple(outputs.shape)}, forward/backward=ok")


def train_model(model_name, seed, config):
    set_seed(seed)
    device = get_device()
    train_loader, val_loader, test_loader = get_elpv_loaders(
        batch_size=config["batch_size"], generator_seed=seed
    )
    model = create_model(model_name=model_name, num_classes=2).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    verify_model_step(model, train_loader, criterion, device)
    artifact_dir = Path("artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    phase_scores = {}

    for phase, epochs in (("head_only", config["phase1_epochs"]), ("finetune", config["phase2_epochs"])):
        if epochs == 0:
            continue
        trainable_names = model.configure_trainable(phase)
        trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        print(f"{model_name} {phase}: {trainable_names}; trainable parameters={trainable_count}")
        optimizer = torch.optim.AdamW(
            parameter_groups(model, phase, config), weight_decay=config["weight_decay"]
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )
        phase_path = artifact_dir / f"{model_name}_{phase}.pth"
        early_stopping = EarlyStopping(
            patience=config["patience"] if phase == "finetune" else epochs + 1,
            min_delta=config["min_delta"], mode="max"
        )
        _, score = fit(
            model, train_loader, val_loader, criterion, optimizer, device, epochs,
            scheduler=scheduler, early_stopping=early_stopping,
            checkpoint_path=phase_path, phase_name=phase,
        )
        phase_scores[phase] = score

    best_phase = max(phase_scores, key=phase_scores.get)
    phase_paths = [artifact_dir / f"{model_name}_{phase}.pth" for phase in phase_scores]
    if len({path.name for path in phase_paths}) != len(phase_paths) or not all(path.is_file() for path in phase_paths):
        raise RuntimeError("Phase checkpoints were not saved as distinct files")
    best_phase_path = artifact_dir / f"{model_name}_{best_phase}.pth"
    best_checkpoint = torch.load(best_phase_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    final_path = artifact_dir / f"best_model_{model_name}.pth"
    torch.save({**best_checkpoint, "model_name": model_name}, final_path)
    metrics = evaluate_with_report(model, test_loader, criterion, device)
    manifest = json.loads(SPLIT_MANIFEST_PATH.read_text(encoding="utf-8"))
    metrics.update({
        "model_name": model_name,
        "class_1_definition": "defectuosa (defect_threshold=0.5)",
        "seed": seed,
        "split_id": manifest["dataset_signature"],
        "best_phase": best_phase,
        "best_epoch": best_checkpoint["best_epoch"],
        "hyperparameters": config,
    })
    save_metrics_report(metrics, model_name=model_name)
    print(f"Selected {best_phase} by validation F1={phase_scores[best_phase]:.4f}; test report saved")


def main():
    parser = argparse.ArgumentParser(description="Reproducible two-phase ELPV training")
    parser.add_argument("--model", choices=("resnet18", "efficientnet_b0", "all"), default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--head-only", action="store_true", help="Run only the phase-1 head baseline")
    args = parser.parse_args()
    config = dict(CONFIG)
    if args.head_only:
        config["phase2_epochs"] = 0
    models = ("resnet18", "efficientnet_b0") if args.model == "all" else (args.model,)
    for model_name in models:
        train_model(model_name, args.seed, config)


if __name__ == "__main__":
    main()