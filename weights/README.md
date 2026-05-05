# Model Weights

Pre-trained model weights for reproducing the results in the paper.

**Status:** Weight distribution is pending approval. Check back for updates or contact the authors.

## Expected Contents

When available, this directory will contain:

```
weights/
  ├── bcnb_2class_ca_genconv_attn.pth    # EP1: Binary CA (GENConv + gated attention)
  ├── bcnb_3class_ca_genconv_attn.pth    # EP1: Ternary CA
  ├── bcnb_4class_ca_genconv_attn.pth    # EP1: Quaternary CA
  ├── bcnb_2class_nca_vgg16_attn.pth     # EP1: Binary NCA (VGG16 + attention MIL)
  ├── bcnb_3class_nca_vgg16_attn.pth     # EP1: Ternary NCA
  └── bcnb_4class_nca_vgg16_attn.pth     # EP1: Quaternary NCA
```

## Retraining from Scratch

All models can be retrained using the scripts in `scripts/training/`. See the main README for instructions.
