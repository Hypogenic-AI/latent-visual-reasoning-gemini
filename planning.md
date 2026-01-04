# Research Plan: Formalizing Latent Visual Reasoning

## Research Question
Can a formal framework based on **object-centric token dynamics** effectively model and predict the behavior of physical systems in a latent space, outperforming unstructured representations?

## Background and Motivation
Current Multimodal LLMs often reason about visual scenes using continuous feature maps or monolithic embeddings. However, human visual reasoning is highly object-centric and causal—we perceive distinct objects and predict their future states based on physical intuitions. "Perception Tokens" and "Slot Attention" offer a way to discrete-ize scenes, but a rigorous framework for their *dynamics* (how they evolve over time) is needed to enable true physical reasoning (e.g., "what happens next?").

## Hypothesis Decomposition
1.  **H1 (Object-Centricity)**: Slot Attention can successfully decompose a dynamic scene (Bouncing Balls) into distinct latent tokens representing individual objects without explicit supervision.
2.  **H2 (Predictable Dynamics)**: The temporal evolution of these object tokens can be modeled by a learned transition function (Dynamics Model) more accurately than pixel-level or monolithic latent dynamics.

## Proposed Methodology

### Approach
We will implement a **Slot-based Video Prediction Framework**:
1.  **Decomposition**: Use **Slot Attention** to extract $N$ object slots from each video frame.
2.  **Dynamics**: Train a **Transformer** (or Interaction Network) to predict the slots of frame $t+1$ given slots from frames $t, t-1, ...$.
3.  **Reconstruction**: Decode predicted slots back to pixels to verify semantic consistency.

We will use a **Synthetic Bouncing Balls** dataset. This is a standard benchmark for unsupervised object-centric learning (used in AIR, S-VAE, Slot Attention papers) and allows for controlled complexity (number of balls, velocity, collisions).

### Experimental Steps
1.  **Data Generation**: Implement a script to generate `Bouncing Balls` videos (colored circles moving with simple reflection physics).
2.  **Baseline Implementation**: Implement a **ConvLSTM** model that operates on the full image frame (unstructured baseline).
3.  **Proposed Implementation**: Implement **Slot Attention + Temporal Transformer**.
    *   Encoder: CNN + Slot Attention -> Slots
    *   Predictor: Transformer over time (per slot)
    *   Decoder: Spatial Broadcast Decoder -> Image
4.  **Training**: Train both models on the synthetic dataset to minimize Next Frame Prediction MSE.
5.  **Evaluation**: Compare reconstruction quality (MSE) and, crucially, qualitative slot decomposition (visualizing attention masks).

### Baselines
*   **ConvLSTM**: Represents the standard "unstructured" deep learning approach to video prediction.

### Evaluation Metrics
*   **MSE (Mean Squared Error)**: Pixel-wise reconstruction error for the predicted next frame.
*   **Visual Inspection**: Attention maps to verify slots track objects.

### Statistical Analysis Plan
*   Compute mean and std dev of MSE over a held-out test set.
*   T-test to compare the performance of the Slot-based model vs. ConvLSTM.

## Expected Outcomes
*   The Slot-based model should achieve lower MSE on long-term predictions (or comparable on short-term) but, more importantly, should explicitly decompose the scene.
*   Attention maps should show slots "locking on" to individual balls and tracking them.

## Timeline and Milestones
*   **Phase 2 (Setup)**: 10 min. Env setup, Data Gen script.
*   **Phase 3 (Implementation)**: 60 min. Model coding (Slots, Predictor, Baseline).
*   **Phase 4 (Experiments)**: 60 min. Training loops (fast training on small synthetic data).
*   **Phase 5 (Analysis)**: 30 min. Aggregating metrics, generating plots.
*   **Phase 6 (Documentation)**: 20 min. Reporting.

## Potential Challenges
*   **Slot Attention Convergence**: Slot Attention can be sensitive to initialization and hyperparameters (learning rate). *Mitigation*: Use established hyperparameters from the original paper/repo.
*   **Training Time**: Video models are slow. *Mitigation*: Use low resolution (64x64), few frames (10), and simple shapes to ensure convergence within the session.

## Success Criteria
*   Successful generation of the dataset.
*   Training curves showing loss convergence for both models.
*   Visual evidence of object discovery (slots = balls).
