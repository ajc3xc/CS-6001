# DNABERT-2 Fine-Tuning for DNA Sequence Error Correction

**Authors:** Luke Hendrickson, Adam Camerer  
**Institution:** Missouri University of Science and Technology, Department of Computer Science

## Overview

This project fine-tunes DNABERT-2 for DNA sequence error correction, adapting it from its original binary classification framework to a multi-label classification task that predicts the correctness of individual bases in a sequence.

## Contents

- `6001_Final_Report.pdf` — Full project report
- `6001_Final_Report.docx` — Word document version

## Approach

- Used a T2T reference genome (chromosome 21) to generate 3.2 million synthetic 150bp sequences with realistic error profiles (10% indel, 1% substitution per base)
- Modified DNABERT-2 architecture: sigmoid activation replacing softmax, multi-hot label encoding, output dimension alignment to 150
- Fixed Triton flash attention matrix transposition bug
- Trained on NVIDIA A100 80GB over ~2 days

## Results

| Metric | Base DNABERT-2 | Fine-Tuned |
|--------|---------------|------------|
| Accuracy | 0.000 | 0.222 |
| Precision | 0.990 | 0.990 |
| Recall | 0.486 | 1.000 |
| F1 Score | 0.651 | 0.995 |

## Dependencies

```
transformers
torch
triton
pandas
numpy
```

## References

DNABERT-2: Zhou et al., 2023. [arxiv.org/abs/2306.15006](https://arxiv.org/abs/2306.15006)
