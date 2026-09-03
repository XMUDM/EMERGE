# EMERGE: Learning High-Resolution Representations of Spatial Multi-Omics via Bidirectional Progressive Evolution


## Repository structure

```
├── main.py            # EMERGE: preprocessing, training, clustering and evaluation
├── metric.py          # Additional metrics (Jaccard, Dice, F-measure, ...)
├── add_graph_noise.py # Spatial-graph noise injection for robustness experiments
├── EMERGE/            # Model implementation
│   ├── model.py       #   Graph encoder
│   ├── EMERGE_pyG.py  #   Training loop (PyTorch Geometric)
│   ├── preprocess.py  #   Preprocessing and spatial / feature graph construction
│   └── utils.py       #   Clustering (mclust / leiden / louvain) and utilities
├── run.sh             # Main experiments on all datasets
├── requirements.txt
└── .gitignore
```

## Data preparation

The code expects a `Data/` directory next to the scripts (not included — see `.gitignore`).
Each dataset lives in its own subfolder, as referenced in `run.sh`:

```
Data/
├── HLN/                                # 10x: human lymph node (RNA + ADT)
├── MISAR_seq_mouse_E15_brain/          # MISAR: RNA + ATAC
├── MISAR_E18.5_mouse_brain/            # MISAR: RNA + ATAC
├── Mouse_Spleen/                       # SPOTS: RNA + ADT
├── Mouse_Thymus/                       # Stereo-CITE-seq: RNA + ADT
├── Dataset8_Mouse_Brain_H3K4me3/       # Spatial-epigenome-transcriptome: RNA + H3K4me3
├── Dataset9_Mouse_Brain_H3K27ac/       # Spatial-epigenome-transcriptome: RNA + H3K27ac
└── Dataset10_Mouse_Brain_H3K27me3/     # Spatial-epigenome-transcriptome: RNA + H3K27me3
```

Ground-truth labels are passed via `--GT_path` (e.g. `Data/HLN/GT_labels.txt`).
Supported `--data_type` values: `10x`, `SPOTS`, `Stereo-CITE-seq`, `MISAR`,
`Spatial-epigenome-transcriptome`.
Please refer to `main.py` for the exact file names expected inside each dataset folder
(e.g. `adata_RNA.h5ad` + `adata_ADT.h5ad` for `10x`).

## Usage

Run the main experiments on all datasets:

```bash
bash run.sh
```

Single run (example, human lymph node):

```bash
python -u main.py \
    --file_fold './Data/HLN' --data_type '10x' --n_clusters 10 \
    --KNN_k 20 --RNA_weight 5 --ADT_weight 5 \
    --GT_path './Data/HLN/GT_labels.txt' \
    --vis_out_path 'results/EMERGE_HLN.png' --txt_out_path 'results/EMERGE_HLN.txt' \
    --save_metrics_path './results/EMERGE_HLN_metrics.txt' --save_path './results/EMERGE_HLN.h5ad' \
    --hard_weight 500 --cl_weight 1
```

Outputs written to `results/`: cluster label assignments (`.txt`), clustering and spatial
smoothness metrics (ARI, NMI, AMI, FMI, V-measure, homogeneity, completeness, F-measure,
Jaccard, per-cluster Moran's I, training time) in `_metrics.txt`, UMAP / spatial
visualisation (`.png`), and the full AnnData object with embeddings (`.h5ad`).

### Spatial graph-noise robustness

`add_graph_noise.py` corrupts the spatial KNN graph by replacing `k_noise` edges per spot
with random non-neighbor spots (fully deterministic given `--noise_seed`). Pass
`--k_noise K` to `main.py` to run on the perturbed graph; the default `--k_noise 0` keeps
the unperturbed graph. The applied `k_noise` value is recorded in the metrics file.

```bash
python -u main.py --file_fold './Data/MISAR_seq_mouse_E15_brain/' --data_type 'MISAR' \
    --n_clusters 12 --KNN_k 20 --RNA_weight 5 --ADT_weight 5 \
    --GT_path './Data/MISAR_seq_mouse_E15_brain/GT_labels.txt' \
    --vis_out_path 'results/EMERGE_MISAR_E15_kn2.png' --txt_out_path 'results/EMERGE_MISAR_E15_kn2.txt' \
    --save_metrics_path './results/EMERGE_MISAR_E15_kn2_metrics.txt' --save_path './results/EMERGE_MISAR_E15_kn2.h5ad' \
    --hard_weight 200 --cl_weight 1 --k_noise 2 --noise_seed 42
```

## Dependencies

```bash
pip install -r requirements.txt
```

Main dependencies: PyTorch, torch-geometric, scanpy, anndata, scikit-learn, esda, libpysal.

Default clustering uses R `mclust` (via `rpy2`); install [R](https://www.r-project.org/)
together with the `mclust` package if you keep `tool = 'mclust'` in `main.py`
(`leiden` / `louvain` are available as alternatives without R).

## Acknowledgements

The model implementation in `EMERGE/` is built on
[SpatialGlue](https://github.com/JinmiaoChenLab/SpatialGlue) (Zhang et al., *Nature Methods* 2024),
substantially modified for this project.
