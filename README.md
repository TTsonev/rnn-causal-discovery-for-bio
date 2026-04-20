![banner](https://i.imgur.com/44br56g.png)

# On Causal Discovery for Deep Learning in a Biological Context (Bachelor Thesis)

Research codebase and thesis resources exploring causal discovery algorithms on the hidden states and outputs of deep learning models trained to predict high/low protein-to-mRNA ratios (PTRs) in different tissues based on cDNA sequences.

This repository was created as part of the bachelor thesis by **Trayan Tsonev** at the **University of Vienna, 2024** (Supervised by Assoz. Prof. Dipl.-Ing. Dr.techn. Sebastian Tschiatschek).

## Project Structure

- `src/`: Core training pipeline, model definition (RNN-based GRU, 4 layers, 32 units), dataset utilities for handling differing length cDNA sequences (including UTR and CDS regions), and callbacks. Built with PyTorch Lightning.
- `notebooks/`: Jupyter notebooks implementing causal discovery algorithms (LiNGAM, DirectLiNGAM, FCI, GES) on the RNN outputs as presented in the report.
  - `casual_learning_4.ipynb`: Analysis dividing the CDS output into 4 segments.
  - `casual_learning_8.ipynb`: Analysis dividing the CDS output into 8 segments.
  - `casual_learning_utr.ipynb`: Analysis incorporating 5' and 3' Untranslated Regions (UTRs) together with the CDS.
- `data/`: Dataset containing cDNA sequences, `.bed` files, and continuous target values (`protein_expression_levels.csv`). Also stores generated observational logs like `rnn_hidden_states_logs.csv` for causal learning.
- `tests/`: Unit tests for core preprocessing behavior and modeling operations.
- `report.pdf`: The complete final thesis report.

## Requirements

- Python 3.10+
- PyTorch-compatible environment

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Running Training

To train the RNN-based model and generate the hidden states log (observational data for the causal inference step):

```bash
PYTHONPATH=src python src/main.py \
  --settings src/config.yaml \
  --checkpoint_path checkpoints
```

Notes:

- The script trains 4 unidirectional GRUs (32 units each).
- The RNN's output is fed to a dense architecture to classify the binary PTRs (high/low).
- A logger function simultaneously splits the RNN output into equal segments, calculating the means for causal inference analysis.

## Causal Discovery Analysis

After training and generating observational data from the model's inner representations, you can run the causal discovery algorithms.
Launch Jupyter and navigate to the `notebooks/` directory to explore causal relationships between sequence sub-regions, tissues, and the model prediction:

```bash
jupyter notebook notebooks/
```

- Each notebook instantiates methods such as `ICA LiNGAM` to identify variables driving the RNN's decisions, revealing, for example, the strong influence of the UTR segments.

## Development Commands

Run tests:

```bash
PYTHONPATH=src pytest
```

Run linting:

```bash
ruff check src tests
```

## Reproducibility

- The configurations for random seed, data splits, and model hyperparameters are defined in `src/config.yaml`.
- Each run writes detailed metadata (including versions and timestamps) to `run_metadata.json` in the respective log directory.

## License

This repository is for academic/research use in the context of a bachelor thesis.
