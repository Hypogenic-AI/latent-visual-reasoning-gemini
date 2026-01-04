# Code Walkthrough

This document provides a high-level overview of the codebase for the "Formalizing Latent Visual Reasoning" project.

## Project Structure

```
.
├── src/                # Source code
│   ├── modules.py      # Core model architectures (Slot Attention, Transformer, Encoder/Decoder)
│   ├── baseline.py     # Baseline ConvLSTM implementation
│   ├── data_gen.py     # Synthetic data generation script
│   ├── train.py        # Main training loop
│   ├── evaluate.py     # Quantitative evaluation script
│   └── ...             # Analysis scripts (ablation, ood, visualization)
├── datasets/           # Generated .pt files
├── results/            # Saved models, plots, and metrics
└── REPORT.md           # Research findings
```

## Key Components (`src/modules.py`)

### 1. `SlotAttention`
*   **Purpose**: Implements the iterative attention mechanism to map `N_inputs` (pixels) to `K` slots.
*   **Key Logic**:
    *   Initialize slots from Gaussian parameters.
    *   Iterate 3 times:
        *   Compute Attention $A = \text{Softmax}(Q(slots) \cdot K(inputs))$.
        *   Update slots using GRU: $Slot_{new} = GRU(Slot_{old}, \text{WeightedMean}(inputs))$.
*   **Input**: Feature map $(B, L, D_{enc})$.
*   **Output**: Latent slots $(B, K, D_{slot})$.

### 2. `SlotPredictor`
*   **Purpose**: Models the physical dynamics in the latent space.
*   **Architecture**:
    *   **Interaction**: A Transformer Encoder (`nn.MultiheadAttention`) allows slots to exchange information (model collisions).
    *   **Dynamics**: A Linear layer (residual) predicts the next state: $S_{t+1} = S_t + \Delta(Transformer(S_t))$.
*   **Key Feature**: Returns attention weights to visualize *which* slots interact.

### 3. `SpatialBroadcastDecoder`
*   **Purpose**: Reconstructs the image from slots.
*   **Method**: Broadcasts each slot vector across a 64x64 grid, adds positional embeddings, and decodes via CNN.
*   **Output**: Reconstructed image and alpha masks (for segmentation).

## Training Pipeline (`src/train.py`)

1.  **Data Loading**: Loads `.pt` files containing video tensors $(B, T, C, H, W)$.
2.  **Model Forward**:
    *   Encoder + Slot Attention $\to$ Extract slots for all frames.
    *   Predictor $\to$ Predict $S_{t+1}$ from $S_t$.
    *   Decoder $\to$ Reconstruct $I_t$ (reconstruction loss) and $I_{t+1}$ (prediction loss).
3.  **Optimization**: Minimizes `MSE(Recon) + MSE(Pred)` using Adam.

## Analysis Scripts

*   `analyze_slots.py`: Checks if slots track stable objects (Center of Mass).
*   `analyze_causality.py`: Perturbs one slot to see if it affects others (Interaction Matrix).
*   `analyze_attention.py`: Plots Transformer attention weights during collisions.
*   `analyze_counterfactual.py`: Simulates "removing" a ball to test causal independence.

## Reproducibility
To reproduce the full suite of experiments:
1.  Generate data: `python src/data_gen.py`
2.  Train: `python -m src.train`
3.  Run all analysis scripts in `src/`.
