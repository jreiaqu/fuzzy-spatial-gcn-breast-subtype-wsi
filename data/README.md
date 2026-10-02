# Data

This directory should contain pre-processed graph data for both datasets.

## BCNB Dataset

The Early Breast Cancer Core-Needle Biopsy (BCNB) dataset is publicly available:
- **Paper:** [Xu et al., Frontiers in Oncology 2021](https://doi.org/10.3389/fonc.2021.759007)
- **Download:** Available through the original publication

Place the pre-processed `.pt` graph files in `data/BCNB/`.

Structure used by the training scripts:
```
data/BCNB/
  ├── ground_truth/patient-clinical-data.xlsx
  ├── patches_paths_class_perc/            # official train/val/test split CSVs
  ├── results_graphs_november_23/          # k-NN graphs from WSI2Graph, one directory per task
  ├── results_graphs_november_23_morph/    # + morphological distance (compute_morphological_edges.py)
  └── results_graphs_november_23_fuzzy/    # rebuilt fuzzy graphs (generate_fuzzy_sigma_grid.py)
```

## SBC Dataset

The Stavanger Breast Cancer (SBC) dataset is a private collection from Stavanger University Hospital, Norway. Access is available under restricted data access agreement. See [Nielsen et al., Cancers 2024](https://doi.org/10.3390/cancers17193234) for cohort details.

Place the pre-processed `.pt` graph files in `data/SBC/`.

Structure used by `scripts/training/retrain_classifier_predictions.py`:
```
data/SBC/
  ├── results_graphs_january_25/<TASK>/graphs_k_19/          # k-NN graphs
  ├── results_graphs_january_25_morph/<TASK>/graphs_k_19/    # + morphological distance
  └── results_graphs_january_25_option1{,_rank2,_rank3}/<TASK>/graphs_k_19/   # rebuilt fuzzy graphs
data/CLARIFY/CLARIFY JANUARY 2024/unified_clinical_info_CBDC_jan2024.xlsx  # SBC ground truth
```
`<TASK>` is OTHERvsTNBC, LUMINALSvsHER2vsTNBC or LUMINALAvsLAUMINALBvsHER2vsTNBC.

## Graph Construction

Graphs are constructed from H&E whole-slide images using the WSI2Graph pipeline:
1. Tissue segmentation and patch extraction at 10x magnification
2. Feature extraction using VGG16 (512-d) or CONCH foundation model (512-d)
3. KNN graph construction (k=19) based on normalized Euclidean distance of patch coordinates

Each `.pt` file contains a PyTorch Geometric `Data` object with:
- `x`: Node features (N x 512)
- `edge_index`: Graph connectivity (2 x E)
- `edge_features`: normalized spatial distance of each edge
- `centroid`: patch coordinates

`compute_morphological_edges.py` adds `x_norm` and `edge_feat_dist`; `generate_fuzzy_graphs.py` adds `edge_index_fuzzy`, `edge_mu_fuzzy` and the related fields (see `scripts/fuzzy/inspect_pt.py`).
