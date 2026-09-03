import os
import scipy
import anndata
import sklearn
import torch
import random
import numpy as np
import scanpy as sc
import pandas as pd
from typing import Optional
import scipy.sparse as sp
from torch.backends import cudnn
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph 
from scipy.sparse import csr_matrix
import torch.nn.functional as F
from scipy.stats import entropy
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity


def construct_neighbor_graph(adata_omics1, adata_omics2, datatype='SPOTS', n_neighbors=6, Arg=None, k_noise=0, noise_seed=42):
    """
    Construct neighbor graphs, including feature graph and spatial graph. 
    Feature graph is based expression data while spatial graph is based on cell/spot spatial coordinates.

    Parameters
    ----------
    n_neighbors : int
        Number of neighbors.

    Returns
    -------
    data : dict
        AnnData objects with preprossed data for different omics.

    """

    # construct spatial neighbor graphs
    ################# spatial graph #################
    if datatype in ['Stereo-CITE-seq', 'Spatial-epigenome-transcriptome']:
       n_neighbors=6 
    # omics1
    cell_position_omics1 = adata_omics1.obsm['spatial']
    cell_position_omics2 = adata_omics2.obsm['spatial'] # should be identical to omics1 
    adj_omics, feature_graph_omics1, feature_graph_omics2, indices = construct_graph_by_coordinate(cell_position_omics1, adata_omics1, adata_omics2, n_neighbors=n_neighbors, k=Arg.KNN_k, n_clusters=Arg.n_clusters, k_noise=k_noise, noise_seed=noise_seed)
    adata_omics1.uns['adj_spatial'] = adj_omics
    adata_omics2.uns['adj_spatial'] = adj_omics
    adata_omics1.obsm['adj_feature'], adata_omics2.obsm['adj_feature'] = feature_graph_omics1, feature_graph_omics2
    adata_omics1.obsm['neibor_indices'] = indices
    adata_omics2.obsm['neibor_indices'] = indices
    data = {'adata_omics1': adata_omics1, 'adata_omics2': adata_omics2}
    
    return data

def pca(adata, use_reps=None, n_comps=10):
    
    """Dimension reduction with PCA algorithm"""
    
    from sklearn.decomposition import PCA
    from scipy.sparse.csc import csc_matrix
    from scipy.sparse.csr import csr_matrix
    pca = PCA(n_components=n_comps)
    if use_reps is not None:
       feat_pca = pca.fit_transform(adata.obsm[use_reps])
    else: 
       if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
          feat_pca = pca.fit_transform(adata.X.toarray()) 
       else:   
          feat_pca = pca.fit_transform(adata.X)
    
    return feat_pca

def clr_normalize_each_cell(adata, inplace=True):
    
    """Normalize count vector for each cell, i.e. for each row of .X"""

    import numpy as np
    import scipy

    def seurat_clr(x):
        # TODO: support sparseness
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()
    
    # apply to dense or sparse matrix, along axis. returns dense matrix
    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.X.A if scipy.sparse.issparse(adata.X) else np.array(adata.X))
    )
    return adata     

def feature_distance(adata_omics1):
    X_1 = adata_omics1.obsm['feat'].copy()
    X_1_mean = np.mean(X_1, axis=1, keepdims=True)
    X_1_std = np.std(X_1, axis=1, keepdims=True)
    X_1_centered = (X_1 - X_1_mean) / X_1_std
    X_1_centered = X_1_centered / np.sqrt(X_1.shape[1])  # scale by feature dimension
    corr_1 = np.dot(X_1_centered, X_1_centered.T)
    # convert correlation to distance
    dist_1 = 1 - corr_1 
    return dist_1

def construct_graph_by_feature(n_samples, dist_1, dist_2, k, adata_omics1, adata_omics2):
    pruned_indices = []
    pruned_values = []
    mean_dist1 = np.mean(dist_1, axis=1)
    mean_dist2 = np.mean(dist_2, axis=1)
    cluster_label_1 = adata_omics1.obs['raw_cluster'].values.to_numpy()
    cluster_label_2 = adata_omics2.obs['raw_cluster'].values.to_numpy()

    for i in range(n_samples):
        nonzero_indices = np.where(dist_1[i] >= 1)[0]
        item_num = len(nonzero_indices)
        
        if item_num <= k+1:
            t_knn_ind = np.argsort(dist_1[i])[:item_num]
            t_value = dist_1[i][t_knn_ind]
        else:
            t_knn_ind = np.argsort(dist_1[i])[:k+1]
            t_value = dist_1[i][t_knn_ind]

        valid_indices = []
        valid_values = []
        for idx, row in enumerate(t_knn_ind): # may keep fewer than K neighbors
            # prior-informed pruning
            if ((dist_1[i, row] < mean_dist1[i] and dist_2[i, row] < mean_dist2[i]) and (cluster_label_1[row] == cluster_label_1[i] and cluster_label_2[row] == cluster_label_2[i])):
                valid_indices.append(row) 
                valid_values.append(1 - dist_1[i, row])

        pruned_indices.append(valid_indices)
        pruned_values.append(valid_values)

    # build adjacency matrix
    x = []
    y = []
    z = []
    for i in range(len(pruned_indices)):
        x.extend([i] * len(pruned_indices[i]))
        y.extend(pruned_indices[i])
        z.extend(pruned_values[i])
    
    index_x = np.array(x)
    index_y = np.array(y)
    values = np.ones(len(x))
    feature_graph_omics1 = csr_matrix((values, (index_x, index_y)), shape=(n_samples, n_samples))
    return feature_graph_omics1


def construct_graph_by_coordinate(cell_position, adata_omics1, adata_omics2, n_neighbors=6, k=20, n_clusters=10, k_noise=0, noise_seed=42):
    """Constructing spatial neighbor graph according to spatial coordinates."""
    # spatial graph
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(cell_position)
    _ , indices = nbrs.kneighbors(cell_position)

    # Graph-noise robustness experiment: replace k_noise of the k spatial KNN
    # edges per spot with random non-neighbor spots (see add_graph_noise.py).
    # k_noise = 0 (default) keeps the unperturbed graph.
    if k_noise > 0:
        from add_graph_noise import apply_noise_to_knn_indices
        indices[:, 1:] = apply_noise_to_knn_indices(cell_position, indices[:, 1:], k_noise, k=n_neighbors, seed=noise_seed)

    pre_labels = 'raw_cluster'
    adata_omics1 = mclust_R(adata_omics1, used_obsm='feat', num_cluster=n_clusters)
    adata_omics1.obs[pre_labels] = adata_omics1.obs['mclust']
    adata_omics2 = mclust_R(adata_omics2, used_obsm='feat', num_cluster=n_clusters)
    adata_omics2.obs[pre_labels] = adata_omics2.obs['mclust']

    # feature similarity
    n_samples = adata_omics1.obsm['feat'].shape[0]
    dist_1 = feature_distance(adata_omics1)
    dist_2 = feature_distance(adata_omics2)
    
    mean_dist1 = np.mean(dist_1, axis=1)
    mean_dist2 = np.mean(dist_2, axis=1)
    feature_graph_omics1 = construct_graph_by_feature(n_samples, dist_1, dist_2, k, adata_omics1, adata_omics2)
    feature_graph_omics2 = construct_graph_by_feature(n_samples, dist_2, dist_1, k, adata_omics1, adata_omics2)

    # spatial graph
    pruned_indices = []
    for i in range(indices.shape[0]):
        valid_indices = []
        for idx, row in enumerate(indices[i]):
            # prior-informed pruning
            if ( (dist_1[i, row] < mean_dist1[i] or dist_2[i, row] < mean_dist2[i])): 
                valid_indices.append(row) 
        pruned_indices.append(valid_indices)

    # build adjacency matrix
    x = []
    y = []
    for i in range(len(pruned_indices)):
        x.extend([i] * len(pruned_indices[i]))
        y.extend(pruned_indices[i])

    adj = pd.DataFrame(columns=['x', 'y', 'value'])
    adj['x'] = x
    adj['y'] = y
    adj['value'] = np.ones(len(x))

    return adj, feature_graph_omics1,feature_graph_omics2, indices

def transform_adjacent_matrix(adjacent):
    n_spot = adjacent['x'].max() + 1
    adj = coo_matrix((adjacent['value'], (adjacent['x'], adjacent['y'])), shape=(n_spot, n_spot))
    return adj

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def mclust_R(adata, num_cluster, used_obsm='feat', random_seed=2024):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames='EEI')

    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata

# ====== Graph preprocessing
def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def adjacent_matrix_preprocessing(adata_omics1, adata_omics2):
    """Converting dense adjacent matrix to sparse adjacent matrix"""
    
    ######################################## construct spatial graph ########################################
    adj_spatial_omics1 = adata_omics1.uns['adj_spatial']
    adj_spatial_omics1 = transform_adjacent_matrix(adj_spatial_omics1)
    adj_spatial_omics2 = adata_omics2.uns['adj_spatial']
    adj_spatial_omics2 = transform_adjacent_matrix(adj_spatial_omics2)
    
    adj_spatial_omics1 = adj_spatial_omics1.toarray()   # To ensure that adjacent matrix is symmetric
    adj_spatial_omics2 = adj_spatial_omics2.toarray()
    
    adj_spatial_omics1 = adj_spatial_omics1 + adj_spatial_omics1.T
    adj_spatial_omics1 = np.where(adj_spatial_omics1>1, 1, adj_spatial_omics1)
    adj_spatial_omics2 = adj_spatial_omics2 + adj_spatial_omics2.T
    adj_spatial_omics2 = np.where(adj_spatial_omics2>1, 1, adj_spatial_omics2)

    print(np.count_nonzero(adj_spatial_omics1))
    print(np.count_nonzero(adj_spatial_omics2))

    # convert dense matrix to sparse matrix
    adj_spatial_omics1 = preprocess_graph(adj_spatial_omics1) # sparse adjacent matrix corresponding to spatial graph
    adj_spatial_omics2 = preprocess_graph(adj_spatial_omics2)
    
    ######################################## construct feature graph ########################################
    adj_feature_omics1 = torch.FloatTensor(adata_omics1.obsm['adj_feature'].copy().toarray())
    adj_feature_omics2 = torch.FloatTensor(adata_omics2.obsm['adj_feature'].copy().toarray())
    
    adj_feature_omics1 = adj_feature_omics1 + adj_feature_omics1.T
    adj_feature_omics1 = np.where(adj_feature_omics1>1, 1, adj_feature_omics1)
    adj_feature_omics2 = adj_feature_omics2 + adj_feature_omics2.T
    adj_feature_omics2 = np.where(adj_feature_omics2>1, 1, adj_feature_omics2)
    print(np.count_nonzero(adj_feature_omics1))
    print(np.count_nonzero(adj_feature_omics2))
    
    # convert dense matrix to sparse matrix
    adj_feature_omics1 = preprocess_graph(adj_feature_omics1) # sparse adjacent matrix corresponding to feature graph
    adj_feature_omics2 = preprocess_graph(adj_feature_omics2)
    
    adj = {'adj_spatial_omics1': adj_spatial_omics1,
           'adj_spatial_omics2': adj_spatial_omics2,
           'adj_feature_omics1': adj_feature_omics1,
           'adj_feature_omics2': adj_feature_omics2,
           }
    
    return adj

def lsi(
        adata: anndata.AnnData, n_components: int = 20,
        use_highly_variable: Optional[bool] = None, **kwargs
       ) -> None:
    r"""
    LSI analysis (following the Seurat v3 approach)
    """
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)
    X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    adata.obsm["X_lsi"] = X_lsi[:,1:]

def tfidf(X):
    r"""
    TF-IDF normalization (following the Seurat v3 approach)
    """
    idf = X.shape[0] / (X.sum(axis=0) + 0.0000000001)
    if scipy.sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf   
    
def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'