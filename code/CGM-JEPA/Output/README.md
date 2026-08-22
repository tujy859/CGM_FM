---
language:
- en
library_name: pytorch
license: mit
pipeline_tag: time-series-forecasting
tags:
- cgm
- continuous-glucose-monitor
- self-supervised-learning
- jepa
- time-series
- masked-prediction
- biosignal
- healthcare
- pretrained-encoder
---

# CGM-JEPA Pretrained Encoders

This repository contains frozen self-supervised encoder weights from the paper [CGM-JEPA: Learning Consistent Continuous Glucose Monitor Representations via Predictive Self-Supervised Pretraining](https://huggingface.co/papers/2605.00933). 

The repo contains the **exact checkpoints used to produce Tables 1–8 of the paper** for both the paper's main contributions (CGM-JEPA, X-CGM-JEPA) and the two re-pretrained baselines (GluFormer, TS2Vec).

- **Code:** [github.com/cruiseresearchgroup/CGM-JEPA](https://github.com/cruiseresearchgroup/CGM-JEPA)
- **Pretraining Dataset:** [`CRUISEResearchGroup/CGM-JEPA-Pretraining`](https://huggingface.co/datasets/CRUISEResearchGroup/CGM-JEPA-Pretraining)
- **Downstream Splits:** [`CRUISEResearchGroup/CGM-JEPA-Downstream`](https://huggingface.co/datasets/CRUISEResearchGroup/CGM-JEPA-Downstream)

## Quick start

```bash
huggingface-cli download CRUISEResearchGroup/CGM-JEPA --local-dir Output
```

Then from the [code repository](https://github.com/cruiseresearchgroup/CGM-JEPA):

```bash
# Reproduce paper Tables 1–6
python scripts/run_all_eval.py
```

The downstream eval will load all four checkpoints automatically from the subdirectories below.

## Layout

```
.
├── cgm_jepa/
│   ├── model.safetensors
│   └── config.json
├── x_cgm_jepa/
│   ├── model.safetensors
│   └── config.json
└── baselines/
    ├── gluformer.pt
    └── ts2vec.pkl
```

`cgm_jepa/` and `x_cgm_jepa/` use the standard `PyTorchModelHubMixin` layout — `model.safetensors` for weights, `config.json` for architecture hyperparameters — so they load via the standard `from_pretrained` one-liner.

## Loading examples

### CGM-JEPA / X-CGM-JEPA — `from_pretrained` one-liner

`Encoder` is a `PyTorchModelHubMixin` subclass, so the architecture hyperparameters and weights load in a single call directly from this repo:

```python
from models.encoder import Encoder

encoder = Encoder.from_pretrained("CRUISEResearchGroup/CGM-JEPA", subfolder="cgm_jepa")
encoder.eval()

# X-CGM-JEPA: same call, different subfolder
encoder_x = Encoder.from_pretrained("CRUISEResearchGroup/CGM-JEPA", subfolder="x_cgm_jepa")
```

### Standalone PyTorch — GluFormer

```python
import torch
import torch.nn as nn
from models.gluformer.gluformer import GluFormer

vocab_size = 278
gluformer = GluFormer(
    vocab_size=vocab_size,
    embed_dim=96,
    nhead=6,
    num_layers=3,
    dim_feedforward=192,
    max_seq_length=25000,
    dropout=0.0,
    pad_token=vocab_size,
)
gluformer.load_state_dict(
    torch.load("Output/baselines/gluformer.pt", map_location="cpu")["encoder"]
)
gluformer.output_head = nn.Identity()   # discard the LM head for embedding extraction
gluformer.eval()
```

## Architectures

### `cgm_jepa` and `x_cgm_jepa`

Both use the same `models.encoder.Encoder` class with identical hyperparameters; only the pretraining objective differs. At downstream / inference time only the temporal encoder is used.

| Field | Value |
|---|---|
| `patch_size` | 12 |
| `encoder_kernel_size` | 3 |
| `encoder_embed_dim` | 96 |
| `encoder_nhead` | 6 |
| `encoder_num_layers` | 3 |

**Input**: a tensor of shape `(B, num_patches, patch_size)` (raw glucose values, z-scored).
**Output**: per-patch embedding of shape `(B, num_patches, embed_dim)`.

## Intended use

- **Frozen feature extraction** from raw CGM windows (24-hour, 5-min sampled, 288 timesteps).
- **Linear-probe evaluation**, especially for the metabolic subphenotyping tasks (IR / β-cell dysfunction) described in the paper.
- **Comparison baseline** for new CGM representation methods.

## Citation

```bibtex
@article{muhammad2026cgm,
  title   = {CGM-JEPA: Learning Consistent Continuous Glucose Monitor Representations via Predictive Self-Supervised Pretraining},
  author  = {Muhammad, Hada Melino and Li, Zechen and Salim, Flora and Metwally, Ahmed A},
  journal = {arXiv preprint arXiv:2605.00933},
  year    = {2026}
}
```

## License
Released under the **MIT license**.