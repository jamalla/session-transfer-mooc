# Session-Based Transfer Learning for MOOC Recommendation

This repository contains code and experiments for exploring transfer learning techniques in the context of session-based recommendation for MOOCs (Massive Open Online Courses). The goal is to leverage large-scale interactions from public e-commerce and book datasets to improve recommendation performance on a smaller, data-scarce MOOC dataset (MARS) using techniques like pretraining, fine-tuning, adapters, and meta-learning (Reptile).

## Project Overview

- **Objective**: Address the cold-start and data scarcity problem in MOOC recommendation.
- **Approach**: 
    - **Pretraining**: Train a session-based recommender (SASRec) on large source datasets.
    - **Transfer**: Fine-tune the pretrained model on the target MARS dataset.
    - **Methods**: Standard Fine-tuning, Adapter Modules, and Meta-Learning (Reptile).

## Datasets

### Source Datasets (Pretraining)
- **YOOCHOOSE**: content from the RecSys Challenge 2015, representing e-commerce clickstreams.
- **Amazon Books**: User-item interaction data from the Amazon dataset.

### Target Dataset
- **MARS**: A MOOC dataset used for evaluating session-based recommendations.

## Repository Structure

### Notebooks

**1. Exploratory Data Analysis (EDA)**
- `01_eda_yoochoose.ipynb`: Load and inspect YOOCHOOSE dataset.
- `02_eda_amazon_books.ipynb`: Load and inspect Amazon Books dataset.
- `03_eda_mars.ipynb`: Load and inspect MARS dataset.

**2. Data Processing**
- `04_session_gap_and_timeline_analysis.ipynb`: Analysis of session temporal gaps.
- `05_sessionize_and_prefix_target.ipynb`: Consolidate interactions into session sequences and generate prefix-target pairs for training.
- `05B_build_tensor_dataset.ipynb`: Prepare PyTorch-compatible TensorDatasets.

**3. Modeling & Transfer Learning**
- `06_build_sasrec_model.ipynb`: Implementation and pretraining of the SASRec (Self-Attentive Sequential Recommendation) model.
- `07_transfer_to_mars.ipynb`: Experiments on transferring the pretrained model to the MARS dataset.
- `08_baselines.ipynb`: Comparison with baseline models.

**4. Advanced Techniques**
- `08_reinit_emb_grid.ipynb`: Grid search experiments on embedding initialization strategies.
- `09_adapters.ipynb`: Implementation of Adapter modules for parameter-efficient transfer learning.
- `10_reptile_meta.ipynb`: Application of the Reptile meta-learning algorithm for finding better model initializations.

### Key Directories
- `src/`: Shared utilities and model definitions.
- `data/`: Raw and processed data storage.
- `models/`: Checkpoints for pretrained and fine-tuned models.

## Setup

```bash
pip install -r requirements.txt
```
