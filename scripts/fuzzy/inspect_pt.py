"""
Inspecciona un archivo .pt de grafo WSI.
Compatible con los tres tipos de grafo del pipeline:

  original  — results_graphs_november_23/
                x, centroid, edge_index, edge_features

  morph     — results_graphs_november_23_morph/
                + x_norm, edge_feat_dist (distancia morfológica sobre aristas espaciales)

  fuzzy     — results_graphs_november_23_fuzzy/
                + x_norm,
                  edge_latent (KNN morfológico), edge_features_latent,
                  edge_index_fuzzy, edge_features_fuzzy_s, edge_features_fuzzy_m, edge_mu_fuzzy

Uso:
    python scripts/fuzzy/inspect_pt.py
    python scripts/fuzzy/inspect_pt.py --pt data/BCNB/.../1000_graph.pt
    python scripts/fuzzy/inspect_pt.py --pt data/BCNB/.../1000_graph.pt --plot
"""

import argparse
import os
import sys

import torch
import numpy as np


SEP = "─" * 70

def section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def tensor_stats(t, label=""):
    if t is None:
        return "  None"
    t = t.float()
    prefix = f"  {label}\n" if label else ""
    return (
        f"{prefix}"
        f"  shape : {tuple(t.shape)}\n"
        f"  dtype : {t.dtype}\n"
        f"  min   : {t.min().item():.4f}\n"
        f"  max   : {t.max().item():.4f}\n"
        f"  mean  : {t.mean().item():.4f}\n"
        f"  std   : {t.std().item():.4f}"
    )


def has(data, name):
    return name in data.keys()


def detect_type(data):
    if has(data, "edge_index_fuzzy"):
        return "fuzzy"
    if has(data, "edge_feat_dist"):
        return "morph"
    return "original"


def show_edge_block(data, edge_index_key, edge_features_key, title, extra_keys=None):
    """Muestra conectividad + stats de aristas para cualquier topología."""
    if not has(data, edge_index_key):
        return
    ei = getattr(data, edge_index_key)
    n_edges = ei.shape[1]
    n_nodes = data.x.shape[0]

    section(title)
    print(f"  shape : {tuple(ei.shape)}  (2 × E)")
    degrees = torch.bincount(ei[0], minlength=n_nodes)
    print(f"  Grado medio : {n_edges / n_nodes:.1f}")
    print(f"  Grado min   : {degrees.min().item()}")
    print(f"  Grado max   : {degrees.max().item()}")
    print(f"\n  Primeras 5 aristas (src → dst):")
    for i in range(min(5, n_edges)):
        src, dst = ei[0, i].item(), ei[1, i].item()
        c_src = data.centroid[src].numpy()
        c_dst = data.centroid[dst].numpy()
        print(f"    arista {i}: nodo {src} {c_src} → nodo {dst} {c_dst}")

    if edge_features_key and has(data, edge_features_key):
        ef = getattr(data, edge_features_key)
        print(f"\n  {edge_features_key}:")
        print(tensor_stats(ef))
        print(f"  Primeras 5 valores:")
        for i in range(min(5, n_edges)):
            print(f"    arista {i}: {ef[i].item():.4f}")

    if extra_keys:
        for key, desc in extra_keys.items():
            if has(data, key):
                t = getattr(data, key)
                print(f"\n  {key}  —  {desc}:")
                print(tensor_stats(t))
                print(f"  Primeras 5 valores:")
                vals = t.float()
                for i in range(min(5, t.shape[0])):
                    print(f"    arista {i}: {vals[i].item():.4f}")


def inspect(pt_path: str, plot: bool = False):

    print(f"\nArchivo: {pt_path}")
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    graph_type = detect_type(data)

    # ── Visión general ────────────────────────────────────────────────────────
    section("VISIÓN GENERAL")
    print(f"  Tipo PyG : {type(data).__name__}")
    print(f"  Variante : {graph_type}")
    print(f"  Campos   : {list(data.keys())}")
    n_nodes = data.x.shape[0]
    n_edges = data.edge_index.shape[1]
    print(f"  Nodos    : {n_nodes}")
    print(f"  Aristas (edge_index) : {n_edges}")
    if has(data, "edge_index_fuzzy"):
        print(f"  Aristas (edge_index_fuzzy) : {data.edge_index_fuzzy.shape[1]}")
    if has(data, "edge_latent") and graph_type == "fuzzy":
        print(f"  Aristas (edge_latent/morph KNN) : {data.edge_latent.shape[1]}")

    # ── x ────────────────────────────────────────────────────────────────────
    section("x  —  features de nodo [N × feat_dim]")
    print(tensor_stats(data.x))
    print(f"\n  Primeras 3 filas (primeros 8 valores):")
    for i in range(min(3, n_nodes)):
        print(f"    nodo {i}: {data.x[i, :8].numpy().round(3)} ...")

    # ── x_norm (morph, fuzzy) ─────────────────────────────────────────────────
    if has(data, "x_norm"):
        section("x_norm  —  features de nodo normalizadas a norma unitaria")
        print(tensor_stats(data.x_norm))
        norms = data.x_norm.float().norm(dim=1)
        print(f"\n  Normas: mean={norms.mean():.6f}  std={norms.std():.6f}  "
              f"min={norms.min():.6f}  max={norms.max():.6f}  (esperado ≈ 1.0)")

    # ── centroid ──────────────────────────────────────────────────────────────
    section("centroid  —  posición espacial (fila, col) en grid de patches")
    print(tensor_stats(data.centroid))
    print(f"\n  Rango fila : [{data.centroid[:,0].min().item():.0f}, {data.centroid[:,0].max().item():.0f}]")
    print(f"  Rango col  : [{data.centroid[:,1].min().item():.0f}, {data.centroid[:,1].max().item():.0f}]")
    print(f"\n  Primeros 5 centroides:")
    for i in range(min(5, n_nodes)):
        print(f"    nodo {i}: fila={data.centroid[i,0].item():.0f}, col={data.centroid[i,1].item():.0f}")

    # ── Topología espacial KNN ────────────────────────────────────────────────
    show_edge_block(
        data, "edge_index", "edge_features",
        "edge_index  —  KNN espacial (top-k por L2 raw)  +  edge_features (d_s anisotrópica)",
    )

    # ── Distancia morfológica sobre aristas espaciales (morph) ────────────────
    if has(data, "edge_feat_dist"):
        section("edge_feat_dist  —  distancia morfológica en aristas espaciales  ||x_norm[i]-x_norm[j]||/2  ∈ [0,1]")
        print(tensor_stats(data.edge_feat_dist))
        out_of_range = ((data.edge_feat_dist < 0) | (data.edge_feat_dist > 1)).sum().item()
        print(f"\n  Valores fuera de [0,1]: {out_of_range}")
        print(f"  Primeras 5 distancias:")
        for i in range(min(5, n_edges)):
            src, dst = data.edge_index[0, i].item(), data.edge_index[1, i].item()
            print(f"    arista {i} (nodo {src}→{dst}): {data.edge_feat_dist[i].item():.4f}")

    # ── edge_latent ───────────────────────────────────────────────────────────
    if has(data, "edge_latent") and graph_type == "fuzzy":
        # En fuzzy: KNN morfológico (índices), con edge_features_latent = d_m
        show_edge_block(
            data, "edge_latent", "edge_features_latent",
            "edge_latent  —  KNN morfológico (top-k por d_m mínima)  +  edge_features_latent (d_m)",
        )

    # ── Topología fuzzy (fuzzy) ───────────────────────────────────────────────
    if has(data, "edge_index_fuzzy"):
        show_edge_block(
            data, "edge_index_fuzzy", "edge_features_fuzzy_s",
            "edge_index_fuzzy  —  KNN fuzzy combinado (top-k por (1-d_s)·(1-d_m))",
            extra_keys={
                "edge_features_fuzzy_m": "d_m de aristas fuzzy ∈ [0,1]",
                "edge_mu_fuzzy":         "peso Gaussiano μ = exp(-d_s²/2σ_s²)·exp(-d_m²/2σ_m²) ∈ [0,1]",
            },
        )

    # ── Campos no reconocidos ─────────────────────────────────────────────────
    known = {
        "x", "x_norm", "centroid",
        "edge_index", "edge_features", "edge_feat_dist",
        "edge_latent", "edge_features_latent",
        "edge_index_fuzzy", "edge_features_fuzzy_s", "edge_features_fuzzy_m", "edge_mu_fuzzy",
    }
    unknown = [k for k in data.keys() if k not in known]
    if unknown:
        section("CAMPOS ADICIONALES (no reconocidos por este script)")
        for k in unknown:
            v = getattr(data, k)
            if isinstance(v, torch.Tensor):
                print(f"  {k}: Tensor {tuple(v.shape)}  dtype={v.dtype}")
            else:
                print(f"  {k}: {type(v).__name__} = {v}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    if plot:
        try:
            import matplotlib.pyplot as plt

            coords = data.centroid.numpy()
            has_fuzzy_topo = has(data, "edge_index_fuzzy")
            has_morph_knn  = has(data, "edge_latent") and graph_type == "fuzzy"

            n_topo = 1 + int(has_morph_knn) + int(has_fuzzy_topo)
            n_hist = 1 + int(has(data, "edge_feat_dist") or has(data, "edge_features_latent") or has(data, "edge_features_fuzzy_m"))
            n_panels = n_topo + n_hist
            fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6))
            ax_idx = 0

            def draw_graph(ax, ei, title, color="steelblue"):
                for e in range(min(ei.shape[1], 3000)):
                    x0, y0 = coords[ei[0, e], 1], -coords[ei[0, e], 0]
                    x1, y1 = coords[ei[1, e], 1], -coords[ei[1, e], 0]
                    ax.plot([x0, x1], [y0, y1], color=color, alpha=0.15, lw=0.4)
                ax.scatter(coords[:, 1], -coords[:, 0], s=8, c="red", zorder=3)
                ax.set_title(title)
                ax.set_xlabel("col")
                ax.set_ylabel("-fila")
                ax.set_aspect("equal")

            draw_graph(axes[ax_idx], data.edge_index.numpy(), "KNN espacial (edge_index)")
            ax_idx += 1

            if has_morph_knn:
                draw_graph(axes[ax_idx], data.edge_latent.numpy(),
                           "KNN morfológico (edge_latent)", color="seagreen")
                ax_idx += 1

            if has_fuzzy_topo:
                draw_graph(axes[ax_idx], data.edge_index_fuzzy.numpy(),
                           "KNN fuzzy combinado (edge_index_fuzzy)", color="darkorange")
                ax_idx += 1

            # Histograma distancia espacial
            ax = axes[ax_idx]; ax_idx += 1
            ax.hist(data.edge_features.numpy(), bins=50,
                    color="steelblue", edgecolor="white", linewidth=0.3)
            ax.set_title("edge_features (d_s espacial)")
            ax.set_xlabel("Distancia espacial normalizada")
            ax.set_ylabel("Nº aristas")

            # Histograma distancia morfológica (la que esté disponible)
            if ax_idx < n_panels:
                ax = axes[ax_idx]
                if has(data, "edge_features_fuzzy_m"):
                    vals = data.edge_features_fuzzy_m.numpy()
                    label = "edge_features_fuzzy_m (d_m aristas fuzzy)"
                elif has(data, "edge_feat_dist"):
                    vals = data.edge_feat_dist.numpy()
                    label = "edge_feat_dist (d_m aristas espaciales)"
                elif has(data, "edge_features_latent"):
                    vals = data.edge_features_latent.numpy()
                    label = "edge_features_latent (d_m KNN morph)"
                else:
                    vals = None
                if vals is not None:
                    ax.hist(vals, bins=50, color="darkorange", edgecolor="white", linewidth=0.3)
                    ax.set_title(label)
                    ax.set_xlabel("Distancia morfológica normalizada")
                    ax.set_ylabel("Nº aristas")

            plt.tight_layout()
            out = pt_path.replace(".pt", "_inspect.png")
            plt.savefig(out, dpi=120)
            print(f"\n  Plot guardado en: {out}")
            plt.show()

        except ImportError:
            print("\n  (matplotlib no disponible, omitiendo plot)")


def find_default_pt(base="data/BCNB"):
    for root, _, files in os.walk(base):
        for f in files:
            if f.endswith(".pt"):
                return os.path.join(root, f)
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspecciona un .pt de grafo WSI")
    parser.add_argument("--pt", type=str, default=None,
                        help="Ruta al archivo .pt (si no se pasa, busca uno automáticamente)")
    parser.add_argument("--plot", action="store_true",
                        help="Genera plots del grafo y distribución de distancias")
    args = parser.parse_args()

    pt_path = args.pt
    if pt_path is None:
        pt_path = find_default_pt()
        if pt_path is None:
            print("No se encontró ningún .pt en data/BCNB. Usa --pt <ruta>.")
            sys.exit(1)

    if not os.path.exists(pt_path):
        print(f"Archivo no encontrado: {pt_path}")
        sys.exit(1)

    inspect(pt_path, plot=args.plot)
