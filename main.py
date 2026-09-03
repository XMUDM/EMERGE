import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"


import torch
import scanpy as sc
import numpy as np
import argparse
import time
import scipy.sparse as sp

from EMERGE.preprocess import fix_seed
from EMERGE.preprocess import clr_normalize_each_cell, pca
from EMERGE.preprocess import construct_neighbor_graph, lsi
from EMERGE.EMERGE_pyG import Train_EMERGE

from EMERGE.utils import clustering

from sklearn.metrics import normalized_mutual_info_score, mutual_info_score, adjusted_mutual_info_score
from sklearn.metrics import v_measure_score, homogeneity_score, completeness_score
from sklearn.metrics import adjusted_rand_score, fowlkes_mallows_score
from metric import jaccard, Dice, F_measure

from libpysal.weights import W
from esda.moran import Moran

def main(args):
    # define device
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # read data
    if args.data_type in ['10x', 'SPOTS', 'Stereo-CITE-seq']:
        adata_omics1 = sc.read_h5ad(os.path.join(args.file_fold, args.rna_file))
        adata_omics2 = sc.read_h5ad(os.path.join(args.file_fold, args.adt_file))
    elif args.data_type == 'Spatial-epigenome-transcriptome':
        adata_omics1 = sc.read_h5ad(args.file_fold + '/adata_RNA.h5ad')
        adata_omics2 = sc.read_h5ad(args.file_fold + '/adata_peaks_normalized.h5ad') 
    elif args.data_type == 'MISAR':
        adata_omics1 = sc.read_h5ad(args.file_fold + '/adata_RNA.h5ad')
        adata_omics2 = sc.read_h5ad(args.file_fold + '/adata_peak.h5ad')

    adata_omics1.var_names_make_unique()
    adata_omics2.var_names_make_unique()

    random_seed = 2024
    fix_seed(random_seed)

    # Preprocess
    if args.data_type == '10x':
        # RNA
        X1 = adata_omics1.X.toarray() if sp.issparse(adata_omics1.X) else np.asarray(adata_omics1.X)
        already_processed = np.nanmin(X1) < 0
        if already_processed:
            adata_omics1.obsm['feat'] = pca(adata_omics1, n_comps=adata_omics2.n_vars - 1)
        else:
            sc.pp.filter_genes(adata_omics1, min_cells=10)
            sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
            sc.pp.normalize_total(adata_omics1, target_sum=1e4)
            sc.pp.log1p(adata_omics1)
            sc.pp.scale(adata_omics1)
            adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
            adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=adata_omics2.n_vars - 1)

        # Protein
        if already_processed:
            adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=adata_omics2.n_vars - 1)
        else:
            adata_omics2 = clr_normalize_each_cell(adata_omics2)
            sc.pp.scale(adata_omics2)
            adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=adata_omics2.n_vars - 1)
        data = construct_neighbor_graph(adata_omics1, adata_omics2, datatype=args.data_type, Arg=args, k_noise=args.k_noise, noise_seed=args.noise_seed)

    elif args.data_type == 'Spatial-epigenome-transcriptome':
        # RNA
        sc.pp.filter_genes(adata_omics1, min_cells=10)
        sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata_omics1, target_sum=1e4)
        sc.pp.log1p(adata_omics1)
        sc.pp.scale(adata_omics1)
        adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
        adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=50)
        
        # ATAC 
        adata_omics2 = adata_omics2[adata_omics1.obs_names].copy() 
        if 'X_lsi' not in adata_omics2.obsm.keys():
            sc.pp.highly_variable_genes(adata_omics2, flavor="seurat_v3", n_top_genes=3000)
            lsi(adata_omics2, use_highly_variable=False, n_components=51)
        adata_omics2.obsm['feat'] = adata_omics2.obsm['X_lsi'].copy()
        data = construct_neighbor_graph(adata_omics1, adata_omics2, datatype=args.data_type, Arg=args, k_noise=args.k_noise, noise_seed=args.noise_seed)

    elif args.data_type == 'SPOTS':
        # RNA
        sc.pp.filter_genes(adata_omics1, min_cells=10)
        sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata_omics1, target_sum=1e4)
        sc.pp.log1p(adata_omics1)
        sc.pp.scale(adata_omics1)
        adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
        adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=adata_omics2.n_vars - 1)
        # Protein
        adata_omics2 = clr_normalize_each_cell(adata_omics2)
        sc.pp.scale(adata_omics2)
        adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=adata_omics2.n_vars - 1)
        data = construct_neighbor_graph(adata_omics1, adata_omics2, datatype=args.data_type, Arg=args, k_noise=args.k_noise, noise_seed=args.noise_seed)
        
    elif args.data_type == 'Stereo-CITE-seq':
        # RNA
        sc.pp.filter_genes(adata_omics1, min_cells=10)
        sc.pp.filter_genes(adata_omics2, min_cells=50)
        adata_omics2 = adata_omics2[adata_omics1.obs_names].copy()
        sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata_omics1, target_sum=1e4)
        sc.pp.log1p(adata_omics1)
        adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
        adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=adata_omics2.n_vars - 1)
        # Protein
        adata_omics2 = clr_normalize_each_cell(adata_omics2)
        adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=adata_omics2.n_vars - 1)
        data = construct_neighbor_graph(adata_omics1, adata_omics2, datatype=args.data_type, Arg=args, k_noise=args.k_noise, noise_seed=args.noise_seed)

    elif args.data_type == 'MISAR':
        # RNA
        sc.pp.filter_genes(adata_omics1, min_cells=10)
        sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata_omics1, target_sum=1e4)
        sc.pp.log1p(adata_omics1)
        sc.pp.scale(adata_omics1)
        adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
        adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=50)
        
        # ATAC 
        adata_omics2 = adata_omics2[adata_omics1.obs_names].copy()  
        if 'X_lsi' not in adata_omics2.obsm.keys():
            sc.pp.highly_variable_genes(adata_omics2, flavor="seurat_v3", n_top_genes=3000)
            lsi(adata_omics2, use_highly_variable=False, n_components=51)
        adata_omics2.obsm['feat'] = adata_omics2.obsm['X_lsi'].copy()
        data = construct_neighbor_graph(adata_omics1, adata_omics2, datatype=args.data_type, Arg=args, k_noise=args.k_noise, noise_seed=args.noise_seed)
    
    else:
        assert 0

    model = Train_EMERGE(data, datatype=args.data_type, device=device, Arg=args)

    start_time = time.time()

    # train model
    output = model.train()
    end_time = time.time()
    print("Training time: ", end_time - start_time)

    # eval 
    adata = adata_omics1.copy()
    adata.obsm['emb_latent_omics1'] = output['emb_latent_omics1'].copy()
    adata.obsm['emb_latent_omics2'] = output['emb_latent_omics2'].copy()
    adata.obsm['EMERGE'] = output['EMERGE'].copy()

    # Clustering
    tool = 'mclust' # mclust, leiden, and louvain
    clustering(adata, key='EMERGE', add_key='EMERGE', n_clusters=args.n_clusters, method=tool, use_pca=True)

    label = adata.obs['EMERGE']
    label_array = np.array(label.copy(), dtype=int)
    l = label.tolist()
    

    # Save results
    output_file = args.txt_out_path
    with open(output_file, 'w') as f:
        for num in l:
            f.write(f"{num}\n")

    Our_list = l

    spatial_adj = adata_omics1.uns['adj_spatial']
    nodes = sorted(set(spatial_adj['x']).union(set(spatial_adj['y'])))
    node2idx = {node: idx for idx, node in enumerate(nodes)}
    neighbors = {i: [] for i in range(len(nodes))}
    weights = {i: [] for i in range(len(nodes))}
    for _, row in spatial_adj.iterrows():
        i = node2idx[row['x']]
        j = node2idx[row['y']]
        val = row['value']

        neighbors[i].append(j)
        weights[i].append(val)

        if i != j:
            neighbors[j].append(i)
            weights[j].append(val)

    w = W(neighbors, weights)

    # Per-cluster Moran's I
    unique_clusters = np.unique(label_array)
    moran_per_cluster = {}
    for c in unique_clusters:
        binary_indicator = (label_array == c).astype(float)
        try:
            mi_c = Moran(binary_indicator, w)
            moran_per_cluster[c] = mi_c.I
        except:
            moran_per_cluster[c] = float('nan')
    mean_cluster_moran = np.nanmean(list(moran_per_cluster.values()))

    # Full metrics if GT is available (either a GT_path file or obs['Ground Truth'])
    gt_array = None
    if args.GT_path is not None and os.path.exists(args.GT_path):
        gt_labels = []
        with open(args.GT_path, 'r') as f:
            for line in f:
                gt_labels.append(line.strip())
        gt_array = np.array(gt_labels)
    elif 'Ground Truth' in adata_omics1.obs.columns:
        gt_labels = adata_omics1.obs['Ground Truth'].astype(str).tolist()
        gt_array = np.array(gt_labels)

    if gt_array is not None:
        pred_array = np.array(Our_list)

        Our_mutual_info = mutual_info_score(gt_array, pred_array)
        Our_nmi = normalized_mutual_info_score(gt_array, pred_array)
        Our_ami = adjusted_mutual_info_score(gt_array, pred_array)
        Our_fmi = fowlkes_mallows_score(gt_array, pred_array)
        Our_ari = adjusted_rand_score(gt_array, pred_array)
        Our_V = v_measure_score(gt_array, pred_array)
        Our_F_measure = F_measure(pred_array, gt_array)
        Our_Jaccard = jaccard(pred_array, gt_array)
        Our_completeness = completeness_score(gt_array, pred_array)
        Our_homogeneity = homogeneity_score(gt_array, pred_array)

        print(f"Mutual_Info={Our_mutual_info:.4f}, NMI={Our_nmi:.4f}, AMI={Our_ami:.4f}, "
              f"FMI={Our_fmi:.4f}, ARI={Our_ari:.4f}, V_measure={Our_V:.4f}")
        print(f"F_measure={Our_F_measure:.4f}, Jaccard={Our_Jaccard:.4f}, "
              f"Completeness={Our_completeness:.4f}, Homogeneity={Our_homogeneity:.4f}")
        print(f"Mean_Cluster_Moran_I={mean_cluster_moran:.4f}")

    # Save metrics
    if hasattr(args, 'save_metrics_path') and args.save_metrics_path is not None:
        os.makedirs(os.path.dirname(args.save_metrics_path) if os.path.dirname(args.save_metrics_path) else '.', exist_ok=True)
        with open(args.save_metrics_path, 'w') as f:
            if gt_array is not None:
                f.write(f"Mutual_Info={Our_mutual_info:.6f}\n")
                f.write(f"NMI={Our_nmi:.6f}\n")
                f.write(f"AMI={Our_ami:.6f}\n")
                f.write(f"FMI={Our_fmi:.6f}\n")
                f.write(f"ARI={Our_ari:.6f}\n")
                f.write(f"V_measure={Our_V:.6f}\n")
                f.write(f"F_measure={Our_F_measure:.6f}\n")
                f.write(f"Jaccard={Our_Jaccard:.6f}\n")
                f.write(f"Completeness={Our_completeness:.6f}\n")
                f.write(f"Homogeneity={Our_homogeneity:.6f}\n")
            f.write(f"Mean_Cluster_Moran_I={mean_cluster_moran:.6f}\n")
            for c, score in sorted(moran_per_cluster.items()):
                f.write(f"Cluster_{int(c)}_Moran_I={score:.6f}\n")
            f.write(f"Train_Time={end_time - start_time:.2f}\n")
            f.write(f"k_noise={args.k_noise}\n")

    # visualization
    if args.data_type in ('Stereo-CITE-seq', 'MISAR'):
        adata.obsm['spatial'][:, 1] = -1 * adata.obsm['spatial'][:, 1]
    elif args.data_type == 'SPOTS':
        # flip the tissue image
        adata.obsm['spatial'] = np.rot90(np.rot90(np.rot90(np.array(adata.obsm['spatial'])).T).T).T
        adata.obsm['spatial'][:, 1] = -1 * adata.obsm['spatial'][:, 1]

    spot_size = 200
    import matplotlib.pyplot as plt
    fig, ax_list = plt.subplots(1, 2, figsize=(8, 3))
    sc.pp.neighbors(adata, use_rep='EMERGE', n_neighbors=500)
    sc.tl.umap(adata)
    sc.pl.umap(adata, color='EMERGE', ax=ax_list[0], title='EMERGE', s=spot_size, show=False)
    sc.pl.embedding(adata, basis='spatial', color='EMERGE', ax=ax_list[1], title='EMERGE', s=spot_size, show=False)
    plt.tight_layout(w_pad=0.3)
    plt.savefig(args.vis_out_path)

    adata.write_h5ad(args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EMERGE: boundary-aware integration of spatial multi-omics data')
    parser.add_argument('--file_fold', type=str,
                        help='Path to data folder')
    parser.add_argument('--rna_file', type=str, default='adata_RNA.h5ad',
                        help='RNA h5ad filename inside file_fold')
    parser.add_argument('--adt_file', type=str, default='adata_ADT.h5ad',
                        help='ADT/protein h5ad filename inside file_fold')
    parser.add_argument('--data_type', type=str,
                        help='data_type')
    parser.add_argument('--n_clusters', type=int,
                        help='n_clusters for clustering')

    parser.add_argument('--KNN_k', type=int, default=20, help='KNN_k')
    parser.add_argument('--RNA_weight', type=float, default=5, help='weight')
    parser.add_argument('--ADT_weight', type=float, default=5, help='weight')
    parser.add_argument('--vis_out_path', type=str, default='results/HLN.png', help='vis_out_path')
    parser.add_argument('--txt_out_path', type=str, default='results/HLN.txt', help='txt_out_path')
    parser.add_argument('--GT_path', default='./HLN/GT_labels.txt', type=str, help='GT_path')
    parser.add_argument('--save_metrics_path', default='./results/HLN_metrics.txt', type=str, help='our_path')
    parser.add_argument('--save_path', default='./results/HLN.h5ad', type=str)
    parser.add_argument('--hard_weight', type=float, default=500, help='weight')
    parser.add_argument('--cl_weight', type=float, default=1, help='weight')
    parser.add_argument('--ot_weight', type=float, default=0.1, help='weight')
    parser.add_argument('--gpu_id', type=str, default='0', help='weight')
    # Graph-noise robustness experiments: --k_noise K > 0 replaces K of the
    # spatial KNN edges per spot with random spots (see add_graph_noise.py).
    # The default --k_noise 0 runs on the unperturbed graph.
    parser.add_argument('--k_noise', type=int, default=0, help='Number of spatial edges to replace per spot (0-6 for K=6)')
    parser.add_argument('--noise_seed', type=int, default=42, help='Random seed for noise injection')
    args = parser.parse_args()

    main(args)
