from __future__ import annotations

import logging
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
from sklearn import metrics
from torch import nn, optim

from bio_constants import TISSUES

LOGFILE_PATH = Path(__file__).resolve().parents[1] / "data" / "rnn_logs.csv"


class RNNModel(pl.LightningModule):
    """PyTorch Lightning model for transcript-level prediction."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dim_latent: int,
        num_rnn_layers: int,
        learning_rate: float,
        loss_function_str: str,
        dim_embedding_base: int = 2,
        dim_embedding_annotation: int = 2,
        dim_tissue_embedding: int = 2,
        bce_linear_dim: int = 64,
        loss_cont_weight: float = 0.05,
        lr_scheduler_mode: str = "min",
        lr_scheduler_factor: float = 0.5,
        lr_scheduler_patience: int = 2,
        lr_scheduler_min_lr: float = 1e-6,
        log_file_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_latent = dim_latent
        self.dim_embedding_base = dim_embedding_base
        self.dim_embedding_annotation = dim_embedding_annotation
        self.dim_tissue_embedding = dim_tissue_embedding
        self.num_rnn_layers = num_rnn_layers
        self.learning_rate = learning_rate
        self.loss_function_str = loss_function_str
        self.bce_linear_dim = bce_linear_dim
        self.loss_cont_weight = loss_cont_weight
        self.lr_scheduler_mode = lr_scheduler_mode
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_min_lr = lr_scheduler_min_lr
        self.max_norm = 2
        self.predict_cont = True  # also predict the continuous target (i.e., PTR value)

        self.training_step_outputs = []
        self.training_step_ys = []
        self.training_step_y_preds = []

        self.valid_step_outputs = []
        self.valid_step_ys = []
        self.valid_step_y_preds = []

        self.save_output = False

        self.num_segments = 4

        self.log_file_path = Path(log_file_path) if log_file_path is not None else LOGFILE_PATH
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)

        # clear logfile to keep test-time logs reproducible per run
        with self.log_file_path.open("w", encoding="utf-8"):
            pass

        # construct model
        self.rnn = nn.GRU(
            3 * self.dim_embedding_base + self.dim_tissue_embedding,
            self.dim_latent,
            num_layers=num_rnn_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.5 if num_rnn_layers > 1 else 0.0,
        )

        # create predictors according to specification in target_names
        # Note: for a bidirectional model, Linear layer dims should be doubled
        if self.loss_function_str == "mse":
            self.predictor = nn.Sequential(
                nn.Linear(dim_latent, dim_latent),
                nn.ELU(),
                nn.Linear(dim_latent, dim_out),
            )
            self.loss_function = nn.MSELoss(reduction="none")
            self.loss_function_cont = None

        elif self.loss_function_str == "bce":
            self.predictor = nn.Sequential(
                nn.Linear(dim_latent, self.bce_linear_dim),
                nn.ELU(),
                nn.Linear(self.bce_linear_dim, dim_out),
            )
            if self.predict_cont:
                self.predictor_cont = nn.Sequential(nn.Linear(dim_latent, dim_out))
                self.loss_function_cont = nn.MSELoss(reduction="none")
            self.loss_function = nn.BCEWithLogitsLoss(reduction="none")

        else:
            raise ValueError(f"Invalid loss: {self.loss_function_str}")

        self.embedding_bases = nn.Embedding(
            num_embeddings=4, embedding_dim=self.dim_embedding_base, max_norm=self.max_norm
        )

        self.embedding_annotation = nn.Embedding(
            num_embeddings=6, embedding_dim=self.dim_embedding_annotation, max_norm=self.max_norm
        )

        self.embedding_tissue = nn.Embedding(
            num_embeddings=len(TISSUES),
            embedding_dim=self.dim_tissue_embedding,
            max_norm=self.max_norm,
        )

    def forward(
        self, batch: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y, l, m, m_targets, meta, y_cont, freq = batch

        x_bases_embedded = self.embedding_bases(x[:, :, 0].long()).reshape(
            x.shape[0], x.shape[1] // 3, -1
        )
        meta_embedded = torch.unsqueeze(self.embedding_tissue(meta), dim=1).repeat(
            1, x.shape[1] // 3, 1
        )
        x_embedded = torch.cat([x_bases_embedded, meta_embedded], dim=2)
        x_packed_squence = nn.utils.rnn.pack_padded_sequence(
            x_embedded,
            list(l.cpu().numpy() // 3),
            batch_first=True,
            enforce_sorted=False,
        )

        rnn_output, _ = self.rnn(x_packed_squence)
        h_unpacked = nn.utils.rnn.unpack_sequence(rnn_output)

        rnn_output = torch.cat([x_i[-1].reshape(1, -1) for x_i in h_unpacked], dim=0)

        y_pred = self.predictor(rnn_output[:, :])

        y_score = torch.sigmoid(y_pred) if self.loss_function_str == "bce" else y_pred
        return rnn_output, y_pred, y_score

    def loss(
        self,
        batch: tuple[torch.Tensor, ...],
        return_y_pred: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, y, _, _, m_targets, _, y_cont, _ = batch

        rnn_output, y_pred, y_score = self.forward(batch)

        loss = self.loss_function(y_pred, y)
        loss = loss * m_targets
        loss = torch.sum(loss) / torch.sum(m_targets)

        if (
            self.predict_cont
            and self.loss_function_str == "bce"
            and self.loss_function_cont is not None
        ):
            y_cont_pred = self.predictor_cont(rnn_output)
            loss_cont = self.loss_function_cont(y_cont_pred, y_cont)
            loss_cont = torch.sum(loss_cont) / torch.numel(loss_cont)
            loss = loss + self.loss_cont_weight * loss_cont

        if return_y_pred:
            return loss, y, y_score

        return loss

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop
        # independent of forward
        loss, y, y_pred = self.loss(batch, return_y_pred=True)

        self.training_step_ys.append(y.detach().cpu().numpy())
        self.training_step_y_preds.append(y_pred.detach().cpu().numpy())

        self.training_step_outputs.append(loss)

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y, y_pred = self.loss(batch, return_y_pred=True)

        self.valid_step_ys.append(y.detach().cpu().numpy())
        self.valid_step_y_preds.append(y_pred.detach().cpu().numpy())

        self.log("val_loss", loss, on_step=True, on_epoch=True)
        self.valid_step_outputs.append(loss.item())
        return loss

    def on_test_epoch_start(self):
        self.ys = []
        self.y_preds = []
        self.y_masks = []
        self.test_logdata = []

        self.save_output = True
        logging.info("Test starts")

    def test_step(self, batch, batch_idx):
        x, y, l, m, m_targets, meta, y_cont, freq = batch

        rnn_output, _, y_pred = self.forward(batch)
        self.ys.append(y.detach().cpu().numpy())
        self.y_preds.append(y_pred.detach().cpu().numpy())
        self.y_masks.append(m_targets.detach().cpu().numpy())

        if self.save_output:
            split_size = (self.dim_latent) // self.num_segments
            sections = torch.split(rnn_output, split_size, dim=1)
            section_means = [torch.mean(section, dim=1).detach().cpu() for section in sections]
            section_means = np.array(section_means).T

            batch_logdata = np.hstack(
                [
                    section_means,
                    y_pred.detach().cpu().numpy().reshape(-1, 1),
                    y.detach().cpu().numpy().reshape(-1, 1),
                    l.detach().cpu().numpy().reshape(-1, 1),
                    meta.detach().cpu().numpy().reshape(-1, 1),
                ]
            )
            self.test_logdata.append(batch_logdata)

    def on_test_epoch_end(self):
        if self.test_logdata:
            all_logdata = np.vstack(self.test_logdata)
            with self.log_file_path.open("a", encoding="utf-8") as f:
                np.savetxt(f, all_logdata, delimiter=",", fmt="%.5f")
            self.test_logdata = []

    def on_validation_epoch_end(self):
        y = np.hstack([y_batch.ravel() for y_batch in self.valid_step_ys])
        y_score = np.hstack([y_pred_batch.ravel() for y_pred_batch in self.valid_step_y_preds])

        try:
            auc = metrics.roc_auc_score(y, y_score)
            logging.info(f"valid_epoch AUC: {auc}")
        except ValueError as exc:
            logging.warning("Could not compute validation AUC: %s", exc)

        self.valid_step_y_preds.clear()
        self.valid_step_ys.clear()

        loss = np.mean(self.valid_step_outputs)
        self.valid_step_outputs.clear()
        logging.info(f"validation_epoch_mean: {loss}")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=self.lr_scheduler_mode,
            factor=self.lr_scheduler_factor,
            patience=self.lr_scheduler_patience,
            min_lr=self.lr_scheduler_min_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
