# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project "Formalizing Latent Visual Reasoning".

### Papers
Total papers downloaded: 5

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Perception Tokens... | Bigverdi et al. | 2024 | papers/2412.03548_perception_tokens.pdf | Visual chain-of-thought tokens. |
| Dynamic Visual Reasoning... | Ding et al. | 2021 | papers/2110.15358_diff_physics.pdf | Differentiable physics framework. |
| Slot Attention... | Locatello et al. | 2020 | papers/2006.15055_slot_attention.pdf | Object-centric slots. |
| Mirage... | Unknown | 2025 | papers/2506.17218_mirage.pdf | Machine mental imagery. |
| Interleaved Latent... | Unknown | 2025 | papers/2512.05665_ilvr.pdf | Interleaved reasoning. |

### Datasets
Total datasets downloaded: 1 (Sample) + Instructions

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| CLEVR | Stanford | ~18GB | VQA | datasets/clevr/ | Instructions provided. Dummy sample created. |
| CLEVRER | MIT | Large | Video Reasoning | datasets/clevrer/ | Instructions provided. |

### Code Repositories
Total repositories cloned: 2

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| Slot Attention | github.com/lucidrains/slot-attention | Implementation of Slot Attention | code/slot-attention/ | PyTorch version. |
| VRDP | github.com/dingmyu/VRDP | Differentiable Physics Reasoning | code/vrdp/ | Official repo. |

### Resource Gathering Notes

#### Search Strategy
- Used Google Search and arXiv API (via wget) to locate papers.
- Verified arXiv IDs for precision.
- Searched HuggingFace for datasets.

#### Challenges Encountered
- **Dataset Access**: CLEVR and CLEVRER are large and not easily "streamable" from HuggingFace due to script policies. Created dummy samples and detailed instructions for manual download.
- **PDF Access**: `pdfplumber` was not available, so extraction was based on search results and general knowledge.

### Recommendations for Experiment Design

1.  **Primary Dataset**: **CLEVRER** is recommended for the full research goals (dynamics), but **CLEVR** is better for initial "latent object" validation.
2.  **Baseline Methods**: Start with **Slot Attention** (unsupervised) to see if it can capture objects in CLEVR. Then try to add a "dynamics" module (simple MLP or physics approximation) to predict future states.
3.  **Code Reuse**: Use `lucidrains/slot-attention` as the backbone. It is clean and extensible.
