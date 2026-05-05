# Data

This directory should contain pre-processed graph data for both datasets.

## BCNB Dataset

The Early Breast Cancer Core-Needle Biopsy (BCNB) dataset is publicly available:
- **Paper:** [Xu et al., Frontiers in Oncology 2021](https://doi.org/10.3389/fonc.2021.759007)
- **Download:** Available through the original publication

Place the pre-processed `.pt` graph files in `data/BCNB/`.

Expected structure:
```
data/BCNB/
  ├── graphs/          # Pre-computed KNN spatial graphs (.pt files)
  ├── splits/          # Train/val/test split CSVs
  └── ground_truth/    # Molecular subtype labels
```

## SBC Dataset

The Stavanger Breast Cancer (SBC) dataset is a private collection from Stavanger University Hospital, Norway. Access is available under restricted data access agreement. See [Nielsen et al., Cancers 2024](https://doi.org/10.3390/cancers17193234) for cohort details.

Place the pre-processed `.pt` graph files in `data/SBC/`.

Expected structure:
```
data/SBC/
  ├── graphs/          # Pre-computed KNN spatial graphs (.pt files)
  ├── folds/           # Cross-validation fold assignments
  └── ground_truth/    # Molecular subtype labels
```

## Graph Construction

Graphs are constructed from H&E whole-slide images using the WSI2Graph pipeline:
1. Tissue segmentation and patch extraction at 10x magnification
2. Feature extraction using VGG16 (512-d) or CONCH foundation model (512-d)
3. KNN graph construction (k=19) based on normalized Euclidean distance of patch coordinates

Each `.pt` file contains a PyTorch Geometric `Data` object with:
- `x`: Node features (N x 512)
- `edge_index`: Graph connectivity (2 x E)
- Spatial coordinates for convex hull and topology analysis
