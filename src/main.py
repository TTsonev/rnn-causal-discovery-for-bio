from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import numpy as np
import torch
import yaml
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger
from sklearn import metrics

from model import RNNModel
from data_utils import PTRSingleTargetDataset, pad_to_same_length

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
DEFAULT_DATA_FILE = ROOT_DIR / "data" / "processed" / "rna_sequences_expressions.pkl"


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int
    batch_size: int
    validation_method: str
    validation_frac_train: float
    num_workers: int


@dataclass(frozen=True)
class ModelConfig:
    dim_latent: int
    num_rnn_layers: int
    dim_tissue_embedding: int
    loss: str
    model_file: str | None
    bce_linear_dim: int
    loss_cont_weight: float


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float
    max_epochs: int
    scheduler_mode: str
    scheduler_factor: float
    scheduler_patience: int
    scheduler_min_lr: float


@dataclass(frozen=True)
class CallbackConfig:
    early_stopping_patience: int


@dataclass(frozen=True)
class TrainConfig:
    data_normalize: bool
    runtime: RuntimeConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    callback: CallbackConfig


def configure_runtime(num_threads: int = 16) -> str:
    """Configure compute runtime and return the accelerator to use."""
    torch.set_num_threads(num_threads)
    if torch.cuda.is_available():
        try:
            torch.cuda.set_per_process_memory_fraction(0.25)
        except RuntimeError:
            logging.warning("Could not set CUDA process memory fraction.")
        return "gpu"
    return "cpu"


class cb_report_error(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        epoch_mean = torch.stack(pl_module.training_step_outputs).mean()
        logging.info("training_epoch_mean: %s", epoch_mean)

        y = np.hstack([y_batch.ravel() for y_batch in pl_module.training_step_ys])
        y_score = np.hstack(
            [y_pred_batch.ravel() for y_pred_batch in pl_module.training_step_y_preds]
        )
        try:
            auc = metrics.roc_auc_score(y, y_score)
            logging.info("training_epoch AUC: %s", auc)
        except ValueError as exc:
            logging.warning("Could not compute training AUC: %s", exc)

        pl_module.training_step_outputs.clear()
        pl_module.training_step_y_preds.clear()
        pl_module.training_step_ys.clear()


def validate_config(model_config: dict[str, Any]) -> None:
    """Validate required config fields before training starts."""
    required_paths = [
        ("setup", "seed"),
        ("setup", "batch_size"),
        ("setup", "validation", "method"),
        ("model", "dim_latent"),
        ("model", "num_rnn_layers"),
        ("model", "dim_tissue_embedding"),
        ("model", "loss"),
        ("optimizer", "learning_rate"),
        ("optimizer", "max_epochs"),
    ]
    for path in required_paths:
        ref: Any = model_config
        for key in path:
            if key not in ref:
                raise ValueError(f"Missing config key: {'.'.join(path)}")
            ref = ref[key]

    if model_config["setup"]["batch_size"] <= 0:
        raise ValueError("setup.batch_size must be positive.")
    if model_config["optimizer"]["max_epochs"] <= 0:
        raise ValueError("optimizer.max_epochs must be positive.")
    frac_train = model_config["setup"]["validation"].get("frac_train", 0.0)
    if not 0.0 < frac_train < 1.0:
        raise ValueError("setup.validation.frac_train must be in (0, 1).")
    if model_config["optimizer"]["learning_rate"] <= 0:
        raise ValueError("optimizer.learning_rate must be positive.")
    if model_config.get("callbacks", {}).get("early_stopping_patience", 1) <= 0:
        raise ValueError("callbacks.early_stopping_patience must be positive.")


def parse_train_config(model_config: dict[str, Any]) -> TrainConfig:
    runtime = RuntimeConfig(
        seed=int(model_config["setup"]["seed"]),
        batch_size=int(model_config["setup"]["batch_size"]),
        validation_method=str(model_config["setup"]["validation"]["method"]),
        validation_frac_train=float(model_config["setup"]["validation"]["frac_train"]),
        num_workers=int(model_config["setup"].get("num_workers", 0)),
    )
    model = ModelConfig(
        dim_latent=int(model_config["model"]["dim_latent"]),
        num_rnn_layers=int(model_config["model"]["num_rnn_layers"]),
        dim_tissue_embedding=int(model_config["model"]["dim_tissue_embedding"]),
        loss=str(model_config["model"]["loss"]),
        model_file=model_config["model"].get("model_file"),
        bce_linear_dim=int(model_config["model"].get("bce_linear_dim", 64)),
        loss_cont_weight=float(model_config["model"].get("loss_cont_weight", 0.05)),
    )
    optimizer = OptimizerConfig(
        learning_rate=float(model_config["optimizer"]["learning_rate"]),
        max_epochs=int(model_config["optimizer"]["max_epochs"]),
        scheduler_mode=str(model_config["optimizer"].get("scheduler_mode", "min")),
        scheduler_factor=float(model_config["optimizer"].get("scheduler_factor", 0.5)),
        scheduler_patience=int(model_config["optimizer"].get("scheduler_patience", 2)),
        scheduler_min_lr=float(model_config["optimizer"].get("scheduler_min_lr", 1e-6)),
    )
    callback = CallbackConfig(
        early_stopping_patience=int(
            model_config.get("callbacks", {}).get("early_stopping_patience", 5)
        ),
    )
    return TrainConfig(
        data_normalize=bool(model_config["data"]["normalize"]),
        runtime=runtime,
        model=model,
        optimizer=optimizer,
        callback=callback,
    )


def make_split_indices(
    n_samples: int,
    frac_train: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    perm = rng.permutation(n_samples)
    split = int(n_samples * frac_train)
    return perm[:split], perm[split:]


def make_dataloader(
    dataset: PTRSingleTargetDataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        dataset,
        collate_fn=pad_to_same_length,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def get_git_commit_hash() -> str:
    """Return current git commit hash or 'unknown' when unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def save_run_metadata(
    log_dir: Path,
    config: TrainConfig,
    raw_config: dict[str, Any],
    settings_path: str,
    data_path: str,
    checkpoint_path: str,
    accelerator: str,
) -> Path:
    """Persist run metadata for reproducibility and auditability."""
    log_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit_hash(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "lightning_version": pl.__version__,
        "accelerator": accelerator,
        "seed": config.runtime.seed,
        "paths": {
            "settings": settings_path,
            "data": data_path,
            "checkpoint": checkpoint_path,
            "log_dir": str(log_dir),
        },
        "effective_config": raw_config,
    }
    output_path = log_dir / "run_metadata.json"
    with output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(metadata, file_handle, indent=2, sort_keys=True)
    return output_path


def train() -> None:
    logging.basicConfig(level=logging.INFO)
    accelerator = configure_runtime()

    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Train RNN model on RNA sequence data.",
    )
    parser.add_argument(
        "-s",
        "--settings",
        help="YAML file containing experiment settings",
        default=str(DEFAULT_CONFIG),
    )
    parser.add_argument(
        "-c",
        "--checkpoint_path",
        help="Path for storing model checkpoints",
        default=str(ROOT_DIR / "checkpoints"),
    )
    parser.add_argument(
        "-d",
        "--data_path",
        help="Path to serialized dataset (.pkl)",
        default=str(DEFAULT_DATA_FILE),
    )
    args = parser.parse_args()

    if not Path(args.settings).exists():
        raise ValueError(f"Settings file {args.settings} does not exist.")

    with open(args.settings, "rt", encoding="utf-8") as f:
        model_config = yaml.safe_load(f)
    validate_config(model_config)
    config = parse_train_config(model_config)

    if config.data_normalize:
        raise ValueError("Data normalization not yet supported.")

    pl.seed_everything(config.runtime.seed, workers=True)
    rng = np.random.default_rng(config.runtime.seed)
    dataset = PTRSingleTargetDataset(filename=args.data_path, discretize=True, only_cds=True)
    n_samples = len(dataset)
    logging.info("Read data contains %s samples.", n_samples)

    if config.runtime.validation_method != "split":
        raise ValueError("The chosen validation type is not implemented/available.")

    train_indices, valid_indices = make_split_indices(
        n_samples, config.runtime.validation_frac_train, rng
    )
    train_dataset = dataset.subDataset(train_indices)
    valid_dataset = dataset.subDataset(valid_indices)

    train_loader = make_dataloader(
        train_dataset,
        batch_size=config.runtime.batch_size,
        num_workers=config.runtime.num_workers,
        shuffle=True,
    )
    valid_loader = make_dataloader(
        valid_dataset,
        batch_size=config.runtime.batch_size,
        num_workers=config.runtime.num_workers,
        shuffle=False,
    )

    model = RNNModel(
        dim_in=2,
        dim_out=1,
        dim_latent=config.model.dim_latent,
        num_rnn_layers=config.model.num_rnn_layers,
        dim_tissue_embedding=config.model.dim_tissue_embedding,
        learning_rate=config.optimizer.learning_rate,
        loss_function_str=config.model.loss,
        bce_linear_dim=config.model.bce_linear_dim,
        loss_cont_weight=config.model.loss_cont_weight,
        lr_scheduler_mode=config.optimizer.scheduler_mode,
        lr_scheduler_factor=config.optimizer.scheduler_factor,
        lr_scheduler_patience=config.optimizer.scheduler_patience,
        lr_scheduler_min_lr=config.optimizer.scheduler_min_lr,
    )

    checkpoint_dir = Path(args.checkpoint_path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    callbacks: list[Callback] = [
        cb_report_error(),
        checkpoint_callback,
        LearningRateMonitor(logging_interval="epoch"),
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=config.callback.early_stopping_patience,
        ),
    ]

    experiment_logger = CSVLogger(
        save_dir=str(checkpoint_dir / "logs"),
        name="rna_rnn",
    )
    metadata_path = save_run_metadata(
        log_dir=Path(experiment_logger.log_dir),
        config=config,
        raw_config=model_config,
        settings_path=args.settings,
        data_path=args.data_path,
        checkpoint_path=args.checkpoint_path,
        accelerator=accelerator,
    )
    logging.info("Saved run metadata to %s", metadata_path)

    trainer = pl.Trainer(
        max_epochs=config.optimizer.max_epochs,
        callbacks=callbacks,
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        deterministic=True,
        logger=experiment_logger,
        default_root_dir=args.checkpoint_path,
        devices=[0] if accelerator == "gpu" else 1,
        accelerator=accelerator,
        gradient_clip_val=1.0,
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    if config.model.model_file:
        trainer.save_checkpoint(config.model.model_file)
        logging.info("Saved final model checkpoint to %s", config.model.model_file)

    logging.info("VALIDATION")
    with torch.no_grad():
        model.eval()
        for data_desc, loader in [("train", train_loader), ("valid", valid_loader)]:
            trainer.test(model, loader)
            y = np.hstack([y_i.ravel() for y_i in model.ys])
            y_score = np.hstack([y_pred_i.ravel() for y_pred_i in model.y_preds])
            y_mask = np.hstack([y_mask_i.ravel() for y_mask_i in model.y_masks])

            y = y[y_mask == 1.0]
            y_score = y_score[y_mask == 1.0]

            if config.model.loss == "mse":
                mse = metrics.mean_squared_error(y, y_score)
                logging.info("%s MSE: %s", data_desc, mse)
            elif config.model.loss == "bce":
                try:
                    auc = metrics.roc_auc_score(y, y_score)
                    logging.info("%s AUC: %s", data_desc, auc)
                except ValueError as exc:
                    logging.warning("Could not compute %s AUC: %s", data_desc, exc)


if __name__ == "__main__":
    train()
