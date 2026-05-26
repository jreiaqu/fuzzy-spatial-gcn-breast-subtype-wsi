# Model Weights

Pre-trained model weights for reproducing the results in the paper.

## Available Weights

### Context-Aware (CA) Models - Evaluation Protocol 1

These are the final CA models reported in Table 2 of the paper (within-domain evaluation on BCNB test set, 218 patients).

| File | Task | Architecture | Test Accuracy | Test F1 |
|------|------|-------------|---------------|---------|
| `bcnb_2class_ca_genconv_attn.pth` | Binary (TNBC vs Other) | GENConv 5L + gated attention | 0.872 | 0.867 |
| `bcnb_3class_ca_genconv_attn.pth` | Ternary (Luminals vs HER2+ vs TNBC) | GENConv 5L + gated attention | 0.764 | 0.764 |
| `bcnb_4class_ca_genconv_attn.pth` | Quaternary (L-A vs L-B vs HER2+ vs TNBC) | GENConv 5L + gated attention | 0.486 | 0.482 |

All CA models use:
- 5 GENConv graph convolutional layers
- Gated attention pooling
- k=19 nearest-neighbor graph construction
- Input: 512-d VGG16 patch features

### Non-Context-Aware (NCA) Models - Evaluation Protocol 1

These are the final NCA models reported in Table 2 of the paper (within-domain evaluation on BCNB test set, 218 patients).

| File | Task | Architecture | Size |
|------|------|-------------|------|
| `bcnb_2class_nca_vgg16_attn.pth` | Binary (TNBC vs Other) | VGG16 backbone + attention MIL | 57 MB |
| `bcnb_3class_nca_vgg16_attn.pth` | Ternary (Luminals vs HER2+ vs TNBC) | VGG16 backbone + attention MIL | 57 MB |
| `bcnb_4class_nca_vgg16_attn.pth` | Quaternary (L-A vs L-B vs HER2+ vs TNBC) | VGG16 backbone + attention MIL | 57 MB |

All NCA models use:
- VGG16 backbone (ImageNet pre-trained, fine-tuned)
- Attention-based MIL aggregation
- Learning rate: 2e-3, optimizer: SGD
- Trained on full BCNB dataset (100 epochs)
- See Table 2 in the paper for per-task performance metrics.

## Usage

```python
import torch
from models import PatchGCN  # or equivalent model class

# Load CA model
model = PatchGCN(num_features=512, num_classes=2, num_layers=5, pooling='attention')
model.load_state_dict(torch.load('weights/bcnb_2class_ca_genconv_attn.pth'))
model.eval()
```

## Retraining from Scratch

All models can be retrained using the scripts in `scripts/training/`. See the main README for instructions.
