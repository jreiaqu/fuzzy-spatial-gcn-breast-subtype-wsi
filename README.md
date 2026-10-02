# Ponderación difusa de aristas en GCNs para la predicción del subtipo molecular de cáncer de mama

Código del Trabajo Fin de Grado. Trabajo de referencia: [cvblab/spatial-gcn-breast-subtype-wsi](https://github.com/cvblab/spatial-gcn-breast-subtype-wsi).

Los dos primeros commits contienen el repositorio de referencia sin modificaciones. Cada uno de los siguientes añade un bloque de cambios, descrito en su mensaje de commit.

## Descripción

El modelo de referencia es una GCN (GENConv con *gated attention pooling*) sobre un grafo k-NN espacial de parches de la WSI. En ese modelo solo se tiene en cuenta la distancia espacial determinista. Este trabajo añade un peso a cada arista en función de dos distancias entre parches vecinos:

- Distancia espacial `d_s`: distancia euclídea entre las coordenadas de los parches, normalizando cada eje por su rango en la WSI. Toma valores en [0, √2].
- Distancia morfológica `d_m = ‖x̂_i − x̂_j‖ / 2`, donde `x̂` es el vector de características normalizado. Toma valores en [0, 1].

Las distancias se transforman en grados de pertenencia mediante una función gaussiana, `μ = exp(−d² / 2σ²)`. Se estudian dos variantes:

- **Opción 1: grafo reconstruido.** La topología se construye tomando, para cada parche, los k vecinos con mayor `(1 − d_s)(1 − d_m)` entre todos los pares. El peso `μ_s · μ_m` se calcula una vez y se guarda en el propio grafo (`edge_mode = fuzzy_combined`).
- **Opción 2: topología heredada.** Se mantiene el grafo k-NN espacial original y el peso se calcula en el `forward` del modelo. Los modos son `spatial` (`1 − d_s`), `morphological` (`1 − d_m`), `spatial_fuzzy` (`μ_s`), `morphological_fuzzy` (`μ_m`) y `combined_fuzzy` (`α · μ_s + (1 − α) · μ_m`).

> **Nota:** en la memoria del TFG la numeración de las opciones es la inversa: la Opción 1 de este repositorio corresponde a la Opción 2 de la memoria, y viceversa. Aquí se usa la del código, que es la de los argumentos y las carpetas de resultados.

σ se calibra fijando el peso r que conserva la arista de distancia mediana: `exp(−mediana² / 2σ²) = r`, con r ∈ {0.1, 0.3, 0.5, 0.7, 0.9}, más el caso σ = mediana (r ≈ 0.607).

Se evalúan los dos protocolos del trabajo de referencia:

- **Protocolo 1:** entrenamiento y evaluación en BCNB, con las tareas de 2, 3 y 4 clases.
- **Protocolo 2:** transferencia de BCNB a SBC. El *backbone* entrenado en BCNB se congela y se reentrena únicamente el clasificador.

## Cambios respecto al trabajo de referencia

**Modelo**

- `models/MIL_models.py`: implementación de la GCN, que en el repositorio de referencia era un *placeholder*, con el parámetro `edge_mode`. Por defecto (`use_edge_features=False`) reproduce la GCN del trabajo de referencia, sin atributos de arista, y carga los pesos de `weights/`. Incluye también las clases del modelo NCA, necesarias para cargar sus pesos.

**Generación de grafos** (`scripts/fuzzy/`, nuevo)

- `compute_morphological_edges.py`: añade la distancia morfológica a los grafos originales (Opción 2).
- `generate_fuzzy_graphs.py`: genera los grafos reconstruidos de la Opción 1 para un par de σ.
- `generate_fuzzy_sigma_grid.py`: genera los grafos de las 36 combinaciones de σ de cada tarea.
- `calculate_sigma.py`: calcula los σ con el criterio de la mediana.
- `inspect_pt.py`: muestra el contenido de un fichero de grafo.

**Entrenamiento y transferencia**

- `scripts/training/retrain_gcn_mccv_generic.py` (modificado): entrenamiento en BCNB con `--fuzzy-option` y `--edge-mode`. Las configuraciones de cada modo (24 por tarea) se generan a partir de las tablas de σ del propio script. Se añaden la tarea de 3 clases, el modo `--mode direct`, la ponderación de clases en la pérdida y el guardado del modelo como `state_dict` + `config`.
- `scripts/training/run_option1_sweep.py` (nuevo): barrido de las 36 combinaciones de σ de la Opción 1, con ejecución en paralelo en GPU.
- `scripts/training/retrain_classifier_predictions.py` (modificado): reconstrucción del *ground truth* de SBC y `--fuzzy-mode` para usar como *backbone* los modelos entrenados con pesos en las aristas.

**Pesos**

- `weights/fuzzy/` (nuevo): los 24 *backbones* entrenados en BCNB que se usan en la transferencia a SBC. Su configuración y sus métricas están en [`weights/fuzzy/README.md`](weights/fuzzy/README.md).

**Otros**

- `requirements.txt`: se añaden `torchvision`, `openpyxl` y `tqdm`.
- `retrain_3class_gcn_mccv.py` se mantiene sin cambios, pero queda sustituido por `retrain_gcn_mccv_generic.py --task 3class`.

El diff completo respecto al trabajo de referencia, sin los pesos, se obtiene con:

```bash
git diff 8161047 HEAD -- . ':!weights'
```

## Uso

Los comandos se ejecutan desde la raíz del repositorio.

**1. Generación de grafos**

```bash
# Opción 2: distancia morfológica sobre los grafos de BCNB y de SBC
python scripts/fuzzy/compute_morphological_edges.py
python scripts/fuzzy/compute_morphological_edges.py \
    --input-dir data/SBC/results_graphs_january_25 \
    --output-dir data/SBC/results_graphs_january_25_morph

# Opción 1: grafos de BCNB para las 36 combinaciones de σ de una tarea
# (sin --task, las tres tareas)
python scripts/fuzzy/generate_fuzzy_sigma_grid.py --task 3class

# Opción 1: grafos de SBC con los σ de un backbone (config del .pth en weights/fuzzy/)
python scripts/fuzzy/generate_fuzzy_graphs.py \
    --input-dir data/SBC/results_graphs_january_25/LUMINALSvsHER2vsTNBC/graphs_k_19 \
    --output-dir data/SBC/results_graphs_january_25_option1/LUMINALSvsHER2vsTNBC/graphs_k_19 \
    --sigma-spatial 0.0959 --sigma-morpho 0.4922
```

**2. Entrenamiento en BCNB (Protocolo 1)**

```bash
# Opción 2: un modo de ponderación (spatial, morphological, spatial_fuzzy,
# morphological_fuzzy o combined_fuzzy)
python scripts/training/retrain_gcn_mccv_generic.py --task 3class \
    --edge-mode spatial_fuzzy --device cuda

# Opción 1: una combinación de σ
python scripts/training/retrain_gcn_mccv_generic.py --task 3class --fuzzy-option 1 \
    --fuzzy-subdir sigmas_med_0.9 --device cuda

# Opción 1: barrido completo y resumen (sweep_top5_<tarea>.csv)
python scripts/training/run_option1_sweep.py --device cuda --parallel 3
python scripts/training/run_option1_sweep.py --summarize
```

Los resultados se guardan en `results/option_1/` y `results/option_2/`. Cada ejecución hace la MCCV de todas las configuraciones del modo, entrena el modelo final con cada una y guarda el `.pth` de la ganadora.

**3. Transferencia a SBC (Protocolo 2)**

```bash
# Backbone de referencia (pesos originales de weights/)
python scripts/training/retrain_classifier_predictions.py --models ca --device cuda

# Backbone entrenado con ponderación de aristas (weights/fuzzy/)
python scripts/training/retrain_classifier_predictions.py --models ca \
    --fuzzy-mode combined_fuzzy --device cuda
```

`--fuzzy-mode` admite los cinco modos de la Opción 2 y `option1`, `option1_rank2` y `option1_rank3`. El clasificador se entrena durante 200 épocas fijas, sin *early stopping*, que es el protocolo utilizado en el TFG. `--patience N` activa el *early stopping* sobre la partición evaluada; como la época se elige con los mismos datos con los que se mide, las cifras que se obtienen así son optimistas. Con `--dry-run` se comprueban las rutas de pesos, grafos y *ground truth* sin entrenar.

## Resultados

F1 ponderada, obtenida con este código y los modelos de `weights/fuzzy/`.

> **Nota sobre las cifras de la memoria del TFG.** Tres valores de estas tablas no coinciden con los de la memoria:
>
> - **SBC, referencia `spatial` (tabla 6.3 de la memoria).** La memoria recoge 0.718 / 0.591 / 0.480, obtenidos con *early stopping* sobre la partición evaluada. Aquí todos los *backbones* se evalúan con el mismo protocolo, 200 épocas fijas sin *early stopping*, con el que la referencia da 0.617 / 0.573 / 0.426.
> - **BCNB, `combined_fuzzy` en 3 y 4 clases (tabla 6.1 de la memoria).** La memoria recoge el test de otra configuración de la rejilla (0.672 y 0.480). Aquí se da el de la configuración seleccionada por validación cruzada, que es la que se usa como *backbone* en SBC (0.689 y 0.474).

> **Los resultados no se reproducirán exactamente.** No se fijan semillas en todo el *pipeline*: en el entrenamiento en BCNB solo se fijan las particiones de la validación cruzada, no la inicialización de la red ni el orden de los datos. Al reentrenar con la misma configuración se obtienen cifras distintas, y las diferencias menores que la desviación típica entre particiones no deben interpretarse como mejoras. La transferencia a SBC con los *backbones* incluidos sí da las mismas cifras, porque parte de pesos y particiones fijos.

**Protocolo 1 (BCNB).** Validación cruzada (media ± desviación típica) y test (evaluación única sobre 218 pacientes).

Opción 2:

| Modo | 2 clases CV | 2 clases test | 3 clases CV | 3 clases test | 4 clases CV | 4 clases test |
|---|---|---|---|---|---|---|
| `spatial` (referencia) | 0.934 ± 0.018 | 0.880 | 0.744 ± 0.022 | 0.634 | 0.537 ± 0.027 | 0.404 |
| `morphological` | 0.935 ± 0.016 | 0.877 | 0.744 ± 0.021 | 0.633 | 0.535 ± 0.031 | 0.466 |
| `spatial_fuzzy` | 0.935 ± 0.017 | 0.868 | 0.748 ± 0.027 | 0.652 | 0.535 ± 0.028 | 0.450 |
| `morphological_fuzzy` | 0.934 ± 0.017 | 0.874 | 0.740 ± 0.026 | 0.688 | 0.536 ± 0.038 | 0.448 |
| `combined_fuzzy` | 0.935 ± 0.018 | 0.876 | 0.746 ± 0.020 | 0.689 | 0.539 ± 0.035 | 0.474 |

Opción 1, tres mejores combinaciones de σ del barrido por F1 de validación cruzada. Los modelos de `weights/fuzzy/` son reentrenamientos de estas combinaciones, con cifras algo distintas (ver su README).

| Tarea | Puesto | `--fuzzy-subdir` | F1 CV | F1 test |
|---|---|---|---|---|
| 2 clases | 1 | `sigmas_0.7_0.1` | 0.942 ± 0.018 | 0.863 |
| 2 clases | 2 | `sigmas_0.5_0.7` | 0.939 ± 0.016 | 0.877 |
| 2 clases | 3 | `sigmas_0.7_0.3` | 0.939 ± 0.015 | 0.865 |
| 3 clases | 1 | `sigmas_med_0.9` | 0.759 ± 0.018 | 0.684 |
| 3 clases | 2 | `sigmas_med_0.7` | 0.758 ± 0.023 | 0.705 |
| 3 clases | 3 | `sigmas_0.7_med` | 0.757 ± 0.016 | 0.644 |
| 4 clases | 1 | `sigmas_med_0.5` | 0.544 ± 0.036 | 0.471 |
| 4 clases | 2 | `sigmas_0.5_0.7` | 0.543 ± 0.029 | 0.465 |
| 4 clases | 3 | `sigmas_0.5_0.1` | 0.542 ± 0.026 | 0.434 |

**Protocolo 2 (BCNB → SBC).** Media ± desviación típica de la validación cruzada (5 particiones × 3 repeticiones) sobre SBC, con el *backbone* indicado por `--fuzzy-mode`.

| `--fuzzy-mode` | 2 clases | 3 clases | 4 clases |
|---|---|---|---|
| `spatial` (referencia) | 0.617 ± 0.087 | 0.573 ± 0.030 | 0.426 ± 0.039 |
| `morphological` | 0.651 ± 0.049 | 0.592 ± 0.041 | 0.442 ± 0.028 |
| `spatial_fuzzy` | 0.634 ± 0.057 | 0.617 ± 0.028 | 0.452 ± 0.040 |
| `morphological_fuzzy` | 0.604 ± 0.074 | 0.570 ± 0.040 | 0.455 ± 0.047 |
| `combined_fuzzy` | 0.653 ± 0.066 | 0.603 ± 0.039 | 0.451 ± 0.041 |
| `option1` | 0.708 ± 0.043 | 0.563 ± 0.039 | 0.437 ± 0.029 |
| `option1_rank2` | 0.702 ± 0.065 | 0.588 ± 0.035 | 0.457 ± 0.038 |
| `option1_rank3` | 0.616 ± 0.102 | 0.590 ± 0.027 | 0.442 ± 0.046 |

En resumen:

- **BCNB.** En la métrica de selección (F1 de validación cruzada), los cinco modos de la Opción 2 quedan a menos de una centésima entre sí, muy por debajo de la desviación típica. En test, las variantes con pertenencia difusa superan a la referencia en 3 y 4 clases (hasta 0.689 frente a 0.634, y 0.474 frente a 0.404), pero se trata de una única evaluación y la validación cruzada no anticipa esas diferencias. La Opción 1 obtiene la mejor validación cruzada, aunque explora más configuraciones, y no mejora en test a la Opción 2.
- **SBC.** La mejor variante difusa supera a la referencia en las tres tareas, y en 4 clases la superan todas. Quedan por debajo `morphological_fuzzy` y `option1_rank3` en 2 clases, y `morphological_fuzzy` y `option1` en 3 clases. Muchas de las diferencias son menores que la desviación típica. La mejor variante cambia con la tarea, y la configuración mejor clasificada en BCNB no es necesariamente la que mejor se transfiere.

El análisis completo está en los capítulos 6 y 7 de la memoria del TFG.

## Datos

Los datos no se incluyen en el repositorio: BCNB es público y SBC es de acceso restringido. Los scripts esperan la siguiente estructura (más detalle en [`data/README.md`](data/README.md)):

```
data/
├── BCNB/
│   ├── ground_truth/patient-clinical-data.xlsx
│   ├── patches_paths_class_perc/{train,val,test}_patches_class_perc_0_tp.csv
│   ├── results_graphs_november_23/          # grafos originales
│   ├── results_graphs_november_23_morph/    # Opción 2
│   └── results_graphs_november_23_fuzzy/    # Opción 1, una carpeta por combinación de σ
├── CLARIFY/
│   └── CLARIFY JANUARY 2024/unified_clinical_info_CBDC_jan2024.xlsx
└── SBC/
    ├── results_graphs_january_25/
    ├── results_graphs_january_25_morph/
    └── results_graphs_january_25_option1{,_rank2,_rank3}/
```

## Consideraciones

- **Ground truth de SBC.** El fichero que usa el trabajo de referencia (`CBDC_4_may2024_gt_extended.xlsx`) no estaba disponible. En su lugar se usa la exportación clínica de CLARIFY de enero de 2024, con dos correcciones:

  - Se excluyen los pacientes sin estado ER/PR/HER2 definido.
  - Los subtipos Luminal A y B se asignan según el umbral MAI < 10, ya que no se dispone de Ki67.

  La cohorte resultante tiene 540 pacientes, frente a 533 en el trabajo de referencia. El procedimiento está documentado en `load_sbc_gt()`.
- **Particiones de SBC.** El fichero de particiones del trabajo de referencia tampoco estaba disponible. Se usa `StratifiedKFold` con una semilla fija por repetición, la misma para todos los *backbones*.
- **Calibración de σ.** La mediana de las distancias se calcula sobre todos los grafos, incluidos los de test. Es una fuga de información pequeña (un escalar por tarea y distancia, sin usar etiquetas) y común a todas las variantes, pero conviene tenerla en cuenta.
