# Literature Review: Formalizing Latent Visual Reasoning

## Research Area Overview
The research focuses on **Latent Visual Reasoning**, a paradigm that aims to equip Multimodal Large Language Models (MLLMs) and visual systems with the ability to reason about the visual world using intermediate, often latent, representations ("tokens") rather than just pixel-level data or static embeddings. Key themes include **Object-Centric Learning** (Slot Attention), **Token Dynamics** (how these tokens evolve and interact), and **Differentiable Physics** (grounding reasoning in physical laws). The goal is to move beyond pattern matching to structured, causal, and dynamic understanding of visual scenes.

## Key Papers

### 1. Perception Tokens Enhance Visual Reasoning in Multimodal Language Models (Bigverdi et al., 2024)
- **Key Contribution**: Introduces "Perception Tokens", a set of intrinsic image representations that act as "visual chain-of-thought" steps.
- **Methodology**: The proposed method, AURORA, uses a Vector Quantized Variational Autoencoder (VQ-VAE) to tokenize intermediate visual representations (like depth maps, segmentation masks, or bounding boxes) into discrete tokens. These tokens are then interleaved with text tokens in the MLLM's context window.
- **Relevance**: Directly addresses the "token dynamics" aspect by formalizing visual reasoning steps as a sequence of discrete tokens, bridging the gap between continuous visual features and discrete language reasoning.

### 2. Dynamic Visual Reasoning by Learning Differentiable Physics Models from Video and Language (Ding et al., 2021)
- **Key Contribution**: Proposes the VRDP (Visual Reasoning with Differentiable Physics) framework.
- **Methodology**: Integrates a visual perception module (to extract objects), a neuro-symbolic concept learner, and a differentiable physics engine (impulse-based rigid-body simulator). It infers physical parameters (mass, restitution) from video and language questions.
- **Relevance**: Provides the theoretical grounding for "physics-aware" tokens. A formal framework for latent visual reasoning must account for the physical rules governing object interactions, which this paper explicitly models.

### 3. Object-Centric Learning with Slot Attention (Locatello et al., 2020)
- **Key Contribution**: The Slot Attention module, a mechanism to bind visual input to a set of discrete "slots" (object files).
- **Methodology**: Uses an iterative attention mechanism where slots compete for explanation of the input features. It is unsupervised and permutation invariant.
- **Relevance**: Foundational for "latent objects". Any theory of latent visual reasoning likely relies on the concept of segregating the scene into discrete, manageable entities (tokens/slots) that can then be reasoned over.

### 4. Mirage: Machine Mental Imagery with Latent Visual Tokens (2025)
- **Key Contribution**: Enables "machine mental imagery" by generating and reasoning with latent visual tokens instead of pixel-space images.
- **Methodology**: Augments VLM decoding to produce latent visual tokens. Uses a two-stage training: joint text-visual supervision (distilling from ground truth image embeddings) followed by text-only supervision with RL.
- **Relevance**: Demonstrates that reasoning can happen purely in the latent space, supporting the hypothesis that a formal framework for "latent token dynamics" is a viable path to improved reasoning without the cost of full image generation.

### 5. Interleaved Latent Visual Reasoning (ILVR) with Selective Perceptual Modeling (2025)
- **Key Contribution**: A framework for interleaving text and latent visual reasoning steps.
- **Methodology**: Uses a Momentum Teacher Model to distill relevant visual features into sparse supervision targets, allowing the model to autonomously generate context-aware visual signals during reasoning.
- **Relevance**: Highlights the temporal/sequential aspect of visual reasoning ("dynamics"). The "selective" aspect is crucial for a theoretical framework—knowing *what* to tokenize and *when*.

## Common Methodologies
- **Tokenization of Vision**: Converting continuous visual signals into discrete units (tokens/slots) using VQ-VAEs (Perception Tokens) or iterative attention (Slot Attention).
- **Neuro-Symbolic / Hybrid Approaches**: Combining neural feature extraction with structured reasoning (Physics Engines, Concept Learners).
- **Interleaved Generation**: Treating visual and textual tokens as a unified sequence, allowing "reasoning" to flow between modalities.

## Standard Baselines
- **End-to-End MLLMs (e.g., LLaVA, GPT-4V)**: Often used as baselines to show that "black box" visual reasoning is less effective than structured/token-based approaches for complex physical or spatial tasks.
- **Program-based approaches (e.g., NS-VQA)**: Earlier neuro-symbolic methods that generate executable code.

## Evaluation Metrics
- **Accuracy on Reasoning Tasks**: CLEVR (counting, logical reasoning), CLEVRER (explanatory/predictive reasoning).
- **Physical Parameter Estimation**: Error metrics for mass, friction, velocity estimation.
- **Generative Quality**: IoU for segmentation, L2 loss for trajectory prediction (in latent or pixel space).

## Gaps and Opportunities
1.  **Lack of Formal Dynamics**: While papers propose *architectures* for tokens, there is limited work on a *mathematical theory* of how these tokens *should* evolve (e.g., obeying conservation laws in latent space).
2.  **Unification**: Slot Attention (unsupervised objects) and Perception Tokens (supervised/distilled features) are separate streams. A unified framework combining object discovery with semantic/physical reasoning is missing.

## Recommendations for Experiment
- **Dataset**: **CLEVRER** (Collision Events for Video Representation and Reasoning) is the ideal testbed as it requires reasoning about dynamics, causality, and counterfactuals, fitting the "Differentiable Physics" + "Token Dynamics" theme perfectly.
- **Baselines**: Slot Attention (for object discovery), standard Transformer (as a non-structured baseline).
- **Focus**: Implement a simplified "Latent Physics Token" model: Extract slots -> Evolve slots using a learned transition function (proxy for diff. physics) -> Decode/Answer.
