from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm


def load_data(
    filename: str | Path, discretize: bool = False
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess sequence data from a pickle file.

    Args:
        filename: Path to the data file.
        discretize: Whether to load binarized targets instead of continuous ones.

    Returns:
        (data_lengths, data_inputs, data_inputs_frequencies, data_targets, data_transcripts).
    """
    with Path(filename).open("rb") as handle:
        data = list(pickle.load(handle).items())

    data_lengths = np.array([len(transcript[1]["fasta"]) for transcript in data])
    data_transcripts = np.array([transcript[0] for transcript in data])

    data_inputs = [transcript[1]["fasta"].reshape(-1, 1) for transcript in data]
    data_inputs2 = [transcript[1]["bed_annotation"].reshape(-1, 1) for transcript in data]
    data_inputs = [
        np.concatenate([in1, in2], axis=1) for in1, in2 in zip(data_inputs, data_inputs2)
    ]

    data_inputs_frequencies = np.vstack(
        [transcript[1]["codon_frequeny"].reshape(1, -1) for transcript in data]
    )

    data_targets = np.vstack([np.expand_dims(transcript[1]["targets"], 0) for transcript in data])

    if discretize:
        logging.info("Discretizing data.")
        data_targets = np.vstack(
            [np.expand_dims(transcript[1]["targets_bin"], 0) for transcript in data]
        )
    else:
        logging.info("NOT discretizing data.")

    return data_lengths, data_inputs, data_inputs_frequencies, data_targets, data_transcripts


def pad_to_same_length(
    batch: list[tuple[Any, ...]],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.LongTensor,
    torch.Tensor,
    torch.Tensor,
    torch.LongTensor,
    torch.Tensor,
    np.ndarray,
]:
    """
    Collate function for data loader - pads data to the same length and
    creates the required masks.
    """
    # inputs, targets, lengths, transcripts, meta, y_cont = batch
    if not batch:
        raise ValueError("Batch must contain at least one sample.")

    l = np.array([s[2] for s in batch]).ravel()
    max_length = np.max(l)

    x = [
        np.pad(
            s[0][0], pad_width=((0, max_length - l[i]), (0, 0)), mode="constant", constant_values=0
        )
        for i, s in enumerate(batch)
    ]
    x = [s.reshape(1, s.shape[0], s.shape[1]) for s in x]
    x = np.vstack(x)

    m = np.ones_like(x)
    for i, length in enumerate(l):
        m[i, length:] = 0

    _x = torch.tensor(x, dtype=torch.float32)
    _m = torch.tensor(m, dtype=torch.float32)

    y = np.vstack([s[1] for s in batch])
    _y = torch.tensor(y, dtype=torch.float32)

    y_cont = np.vstack([s[5] for s in batch])
    _y_cont = torch.tensor(y_cont, dtype=torch.float32)

    _l = torch.LongTensor(l)

    _m_targets = (~torch.isnan(_y)).to(dtype=torch.float32)
    _y[torch.isnan(_y)] = 0

    meta = torch.LongTensor(np.array([s[4] for s in batch], dtype=int).ravel())
    freq = np.array([s[6] for s in batch])
    return _x, _y, _l, _m, _m_targets, meta, _y_cont, freq


class PTRSingleTargetDataset(torch.utils.data.Dataset):
    """
    Dataset for protein-to-mRNA (PTR) ratio predictions.

    Converts the structured sequence data into a typical dataset format with
    a single target per sample (tissue-specific).
    """

    def __init__(self, **args: Any) -> None:
        if "discretize" not in args:
            raise ValueError(
                "Dataset must be discretized, otherwise it is too big to process in this way."
            )
        if not isinstance(args["discretize"], bool):
            raise TypeError("'discretize' must be bool.")
        if not args["discretize"]:
            raise ValueError("PTRSingleTargetDataset currently requires discretize=True.")
        self.discretize = args["discretize"]

        if "only_cds" in args:
            if not isinstance(args["only_cds"], bool):
                raise TypeError("'only_cds' must be bool.")
            self.only_cds = args["only_cds"]
        else:
            self.only_cds = False

        if "filename" in args:
            if not isinstance(args["filename"], str | Path):
                raise TypeError("'filename' must be str or pathlib.Path.")

            # load data from file
            logging.info(f"Loading data from {args['filename']}.")
            (
                self.data_lengths,
                self.data_inputs,
                self.data_inputs_frequencies,
                self.data_targets,
                self.data_transcripts,
                self.data_meta,
                self.data_targets_cont,
            ) = self._load_data(args["filename"])
        else:
            (
                self.data_lengths,
                self.data_inputs,
                self.data_inputs_frequencies,
                self.data_targets,
                self.data_transcripts,
                self.data_meta,
                self.data_targets_cont,
            ) = None, None, None, None, None, None, None

    def _load_data(
        self, filename: str | Path
    ) -> tuple[
        np.ndarray, list[np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """
        Load dataset inputs and targets from a pickle file and separate per tissue.

        Returns:
            Tuple containing arrays and lists of lengths, inputs, frequencies,
            binary targets, transcript names, tissue metadata, and continuous targets.
        """
        with Path(filename).open("rb") as handle:
            data = pickle.load(handle)

        (
            data_lengths,
            data_inputs,
            data_inputs_frequencies,
            data_meta,
            data_targets,
            data_transcripts,
            data_targets_cont,
        ) = [], [], [], [], [], [], []
        for key, transcript in tqdm(data.items()):
            for tissue_idx in np.where(np.logical_not(np.isnan(transcript["targets_bin"])))[0]:
                if self.only_cds:
                    # only consider codon region
                    idx = [t[3] for t in transcript["bed"]].index("CDS")
                    start = int(transcript["bed"][idx][1])
                    end = int(transcript["bed"][idx][2])

                    in1 = transcript["fasta"][start:end].reshape(-1, 1)
                    in2 = transcript["bed_annotation"][start:end].reshape(-1, 1)
                    inc = np.concatenate([in1, in2], axis=1)
                else:
                    in1 = transcript["fasta"].reshape(-1, 1)
                    in2 = transcript["bed_annotation"].reshape(-1, 1)
                    inc = np.concatenate([in1, in2], axis=1)
                data_inputs.append(inc)

                data_lengths.append(len(inc))

                data_inputs_frequencies.append(transcript["codon_frequeny"].reshape(1, -1))

                data_meta.append(tissue_idx)

                data_targets.append(transcript["targets_bin"][tissue_idx])
                data_targets_cont.append(transcript["targets"][tissue_idx])

                data_transcripts.append(key)

        data_lengths = np.array(data_lengths, dtype=int)
        data_inputs_frequencies = np.vstack(data_inputs_frequencies)
        data_targets = np.array(data_targets)
        data_targets_cont = np.array(data_targets_cont)
        data_meta = np.array(data_meta)
        data_transcripts = np.array(data_transcripts)

        return (
            data_lengths,
            data_inputs,
            data_inputs_frequencies,
            data_targets,
            data_transcripts,
            data_meta,
            data_targets_cont,
        )

    def subDataset(self, indices: np.ndarray) -> "PTRSingleTargetDataset":
        """
        Returns a dataset consisting only of a subset of the data as indexed
        by 'indices'.
        """
        dataset = PTRSingleTargetDataset(discretize=True)

        dataset.data_lengths = self.data_lengths[indices]
        dataset.data_inputs = [self.data_inputs[i] for i in indices]
        dataset.data_targets = self.data_targets[indices]
        dataset.data_transcripts = self.data_transcripts[indices]
        dataset.data_inputs_frequencies = self.data_inputs_frequencies[indices]
        dataset.data_meta = self.data_meta[indices]
        dataset.data_targets_cont = self.data_targets_cont[indices]

        return dataset

    def __len__(self) -> int:
        return len(self.data_lengths)

    def __getitem__(
        self, idx: int | list[int] | torch.Tensor
    ) -> tuple[
        list[np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if not isinstance(idx, list):
            idx = [idx]

        lengths = self.data_lengths[idx]
        inputs = [self.data_inputs[i] for i in idx]
        targets = self.data_targets[idx]
        targets_cont = self.data_targets_cont[idx]
        transcripts = self.data_transcripts[idx]
        meta = self.data_meta[idx]

        frequencies = self.data_inputs_frequencies[idx]

        return inputs, targets, lengths, transcripts, meta, targets_cont, frequencies
