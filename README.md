# QM9 Molecular Property Prediction: GCN vs. DimeNet++

This repository contains the codebase for the Course Mini-Project for **24-788 Introduction to Deep Learning (Spring 2026)**. The project focuses on predicting the HOMO-LUMO gap ($\Delta \epsilon$) of small organic molecules using the QM9 dataset.

## Project Overview

Quantum-mechanical properties of molecules are essential for drug discovery and materials science but are computationally expensive to compute via Density Functional Theory (DFT). This project implements and compares two graph-based neural architectures to serve as surrogate models for these properties:

1.  **Baseline Model**: Graph Convolutional Network (GCN) - A standard architecture covered in the course.
2.  **Model Variant**: DimeNet++ - A directional message-passing algorithm that incorporates 3D geometric information (bond lengths and angles), which is not explicitly covered in the course.

### Dataset
We use the **QM9 dataset**, which consists of ~130,000 small organic molecules. Our target is index 4: the HOMO-LUMO gap.

## Environment Setup

This project uses `uv` for Python package management.

### Prerequisites
- Python >= 3.12
- `uv` installed ([Installation Guide](https://docs.astral.sh/uv/getting-started/installation/))

### Installation
1. Clone the repository.
2. Sync the environment:
   ```bash
   uv sync
   ```
3. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

### Dependencies
Key libraries include:
- `torch` & `torch-geometric`: For model implementation and dataset handling.
- `rdkit`: For processing raw molecular data.
- `matplotlib` & `tqdm`: For visualization and progress tracking.

## Dataset Preparation

The `torch_geometric` library handles downloading and preprocessing the QM9 dataset automatically. 

**Note on Data Integrity**: During development, we identified 41 corrupted molecules in the raw QM9 SDF file (indices 4805-4845) that cause parsing errors in RDKit. We have updated `data/QM9/raw/uncharacterized.txt` to include these indices, ensuring they are skipped during processing.

## How to Reproduce Results

### 1. Training
To train the models from scratch, you can use the following notebooks:
- `dimenet.ipynb`: Contains the GCN baseline implementation and initial DimeNet experiments.
- `dimenet2.ipynb`: Contains the optimized DimeNet++ training loop configured for CPU/GPU.

Training will generate:
- `dimenet_best_model.pt`: The trained model weights.
- `dimenet_config.json`: The hyperparameters used for training.

### 2. Evaluation
To reproduce the key metrics reported in the project report without retraining:
1. Ensure `dimenet_best_model.pt` and `dimenet_config.json` are in the root directory.
2. Run the reproduction notebook:
   ```bash
   jupyter notebook reproduce_results.ipynb
   ```
This notebook will load the best checkpoint and evaluate it on the QM9 test set (indices 120,000+).

## Project Structure
- `dimenet2.ipynb`: Main training notebook for the DimeNet++ variant.
- `reproduce_results.ipynb`: Script to regenerate test metrics.
- `data/`: Directory for QM9 raw and processed data.
- `pyproject.toml`: Environment and dependency definition.
