# Backbones entrenados en BCNB con ponderación difusa de aristas

Modelos GCN (GENConv, 5 capas, *gated attention pooling*) entrenados en BCNB con `retrain_gcn_mccv_generic.py`. Son los *backbones* congelados que usa `retrain_classifier_predictions.py --fuzzy-mode <modo>` en la transferencia a SBC.

Cada fichero `bcnb_<tarea>_<modo>.pth` guarda `{'state_dict', 'config'}`. En `config` están `edge_mode` y los parámetros que usa ese modo (σ espacial, σ morfológico, α); los que no usa valen `None`. Se cargan con `load_fuzzy_gcn_model()` de `retrain_classifier_predictions.py`.

## Opción 2 (topología k-NN heredada)

Configuración ganadora de la validación cruzada (MCCV 5 × 3) en cada modo. Las métricas son F1 ponderada en BCNB: media ± desviación típica de la MCCV y evaluación única sobre los 218 pacientes de test.

| `--fuzzy-mode` | Tarea | σ espacial | σ morfológico | α | F1 CV | F1 test |
|---|---|---|---|---|---|---|
| `spatial` | 2 clases | | | | 0.934 ± 0.018 | 0.880 |
| `spatial` | 3 clases | | | | 0.744 ± 0.022 | 0.634 |
| `spatial` | 4 clases | | | | 0.537 ± 0.027 | 0.404 |
| `morphological` | 2 clases | | | | 0.935 ± 0.016 | 0.877 |
| `morphological` | 3 clases | | | | 0.744 ± 0.021 | 0.633 |
| `morphological` | 4 clases | | | | 0.535 ± 0.031 | 0.466 |
| `spatial_fuzzy` | 2 clases | 0.0846 (r = 0.7) | | | 0.935 ± 0.017 | 0.868 |
| `spatial_fuzzy` | 3 clases | 0.0460 (r = 0.3) | | | 0.748 ± 0.027 | 0.652 |
| `spatial_fuzzy` | 4 clases | 0.0714 (mediana) | | | 0.535 ± 0.028 | 0.450 |
| `morphological_fuzzy` | 2 clases | | 0.1579 (r = 0.1) | | 0.934 ± 0.017 | 0.874 |
| `morphological_fuzzy` | 3 clases | | 0.6477 (r = 0.9) | | 0.740 ± 0.026 | 0.688 |
| `morphological_fuzzy` | 4 clases | | 0.1884 (r = 0.3) | | 0.536 ± 0.038 | 0.448 |
| `combined_fuzzy` | 2 clases | 0.0846 | 0.1579 | 0.5 | 0.935 ± 0.018 | 0.876 |
| `combined_fuzzy` | 3 clases | 0.0714 | 0.2973 | 0.7 | 0.746 ± 0.020 | 0.689 |
| `combined_fuzzy` | 4 clases | 0.0714 | 0.2924 | 0.5 | 0.539 ± 0.035 | 0.474 |

## Opción 1 (grafo reconstruido)

Las tres mejores combinaciones de σ del barrido por F1 de validación cruzada. El barrido se ejecutó sin guardar modelos, así que estos ficheros son reentrenamientos de esas combinaciones. Como no hay semillas fijadas, sus cifras difieren algo de las del barrido, que son las de la tabla 6.2 de la memoria.

| `--fuzzy-mode` | Tarea | Grafos (`--fuzzy-subdir`) | σ espacial | σ morfológico | F1 CV | F1 test |
|---|---|---|---|---|---|---|
| `option1` | 2 clases | `sigmas_0.7_0.1` | 0.1151 | 0.1241 | 0.938 ± 0.016 | 0.862 |
| `option1_rank2` | 2 clases | `sigmas_0.5_0.7` | 0.0825 | 0.3153 | 0.937 ± 0.018 | 0.863 |
| `option1_rank3` | 2 clases | `sigmas_0.7_0.3` | 0.1151 | 0.1716 | 0.939 ± 0.015 | 0.874 |
| `option1` | 3 clases | `sigmas_med_0.9` | 0.0959 | 0.4922 | 0.758 ± 0.018 | 0.675 |
| `option1_rank2` | 3 clases | `sigmas_med_0.7` | 0.0959 | 0.2675 | 0.751 ± 0.022 | 0.686 |
| `option1_rank3` | 3 clases | `sigmas_0.7_med` | 0.1136 | 0.2259 | 0.749 ± 0.017 | 0.689 |
| `option1` | 4 clases | `sigmas_med_0.5` | 0.0949 | 0.1927 | 0.535 ± 0.032 | 0.514 |
| `option1_rank2` | 4 clases | `sigmas_0.5_0.7` | 0.0806 | 0.2687 | 0.533 ± 0.034 | 0.447 |
| `option1_rank3` | 4 clases | `sigmas_0.5_0.1` | 0.0806 | 0.1057 | 0.543 ± 0.032 | 0.429 |

Para transferir estos modelos a SBC, los grafos de SBC tienen que generarse con los mismos σ (están en `config['sigma_spatial']` y `config['sigma_morphological']`); ver el README principal.
