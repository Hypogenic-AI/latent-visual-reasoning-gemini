# Formalizing Latent Visual Reasoning

## Project Overview
This project investigates a theoretical framework for **Latent Visual Reasoning** by modeling the dynamics of object-centric "perception tokens". We compare a structured **Slot Attention + Dynamics** model against an unstructured baseline on a video prediction task.

## Key Findings
*   **Object-Centricity Matters**: Decomposing scenes into discrete slots improves predictive modeling.
*   **35% Error Reduction**: The proposed Slot-based model achieved an MSE of **0.0134** compared to **0.0206** for the baseline.
*   **Formal Dynamics**: Explicitly modeling the evolution of tokens ($S_t \to S_{t+1}$) is a viable path for physical reasoning in AI.

## Reproducibility

### Environment Setup
1.  Initialize the environment:
    ```bash
    uv venv
    source .venv/bin/activate
    uv add torch torchvision numpy matplotlib imageio tqdm
    ```

### Execution
1.  **Generate Data**:
    ```bash
    python src/data_gen.py
    ```
2.  **Train Models**:
    ```bash
    python -m src.train
    ```
3.  **Evaluate**:
    ```bash
    python -m src.evaluate
    ```
4.  **Visualize**:
    ```bash
    python -m src.visualize_gif
    ```
    This generates `results/plots/comparison.gif` showing Ground Truth (Top), Baseline (Middle), and Slot Model (Bottom).

## File Structure
*   `src/`: Source code for models, training, and evaluation.
    *   `modules.py`: Implementation of Slot Attention and Dynamics Model.
    *   `baseline.py`: Baseline ConvPredictor.
    *   `train.py`: Main training loop.
    *   `evaluate.py`: Evaluation script.
*   `datasets/`: Data directory.
*   `results/`: Saved models, metrics, and plots.
*   `REPORT.md`: Full research report.

## References
*   Locatello et al., "Object-Centric Learning with Slot Attention", NeurIPS 2020.
*   Ding et al., "Dynamic Visual Reasoning by Learning Differentiable Physics Models", NeurIPS 2021.
