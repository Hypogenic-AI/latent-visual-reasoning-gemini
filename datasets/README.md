# Downloaded Datasets

This directory contains datasets for the research project. Data files are NOT committed to git due to size.

## Dataset 1: CLEVR (Compositional Language and Elementary Visual Reasoning)

- **Source**: [Stanford CLEVR Website](https://cs.stanford.edu/people/jcjohns/clevr/)
- **Size**: ~18 GB
- **Format**: Images (PNG) and Questions (JSON)
- **Task**: Visual Question Answering, Object Discovery
- **License**: CC-BY 4.0

### Download Instructions

```bash
# Download main dataset
wget https://cs.stanford.edu/people/jcjohns/clevr/clevr_v1.0.zip
unzip clevr_v1.0.zip -d datasets/clevr/
```

### Sample Data
See `datasets/clevr/sample.json` for the data structure.

## Dataset 2: CLEVRER (Collision Events for Video Representation and Reasoning)

- **Source**: [CLEVRER Website](http://clevrer.csail.mit.edu/)
- **Size**: Video dataset (Large)
- **Format**: Videos (MP4) and Questions (JSON)
- **Task**: Dynamic Visual Reasoning (Descriptive, Explanatory, Predictive, Counterfactual)

### Download Instructions

```bash
# Visit http://clevrer.csail.mit.edu/ to request access or find download links.
# Typically requires filling a form or direct download if available.
```
