import torch
from tqdm import tqdm
import torch.nn.functional as F
from .model import Encoder_overall
from .preprocess import adjacent_matrix_preprocessing
from .preprocess import mclust_R

from .utils import OT


class Train_EMERGE:
    def __init__(self,
        data,
        datatype = 'SPOTS',
        device= torch.device('cpu'),
        random_seed = 2025,
        learning_rate=0.001,
        weight_decay=0.00,
        epochs=600,
        dim_input=3000,
        dim_output=64,
        weight_factors = [1, 5, 1, 1],
        Arg=None
        ):
        self.data = data.copy()
        self.datatype = datatype
        self.device = device
        self.random_seed = random_seed
        self.learning_rate=learning_rate
        self.weight_decay=weight_decay
        self.epochs=epochs
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.weight_factors = weight_factors

        # adj
        self.adata_omics1 = self.data['adata_omics1']
        self.adata_omics2 = self.data['adata_omics2']

        self.adj = adjacent_matrix_preprocessing(self.adata_omics1, self.adata_omics2)
        self.adj_spatial_omics1 = self.adj['adj_spatial_omics1'].to(self.device)
        self.adj_spatial_omics2 = self.adj['adj_spatial_omics2'].to(self.device)
        self.adj_feature_omics1 = self.adj['adj_feature_omics1'].to(self.device)
        self.adj_feature_omics2 = self.adj['adj_feature_omics2'].to(self.device)

        # feature
        self.features_omics1 = torch.FloatTensor(self.adata_omics1.obsm['feat'].copy()).to(self.device)
        self.features_omics2 = torch.FloatTensor(self.adata_omics2.obsm['feat'].copy()).to(self.device)

        self.n_cell_omics1 = self.adata_omics1.n_obs # number of spots
        self.n_cell_omics2 = self.adata_omics2.n_obs

        self.k = Arg.KNN_k
        self.n_clusters = Arg.n_clusters

        # dimension of input feature
        self.dim_input1 = self.features_omics1.shape[1]
        self.dim_input2 = self.features_omics2.shape[1]
        self.dim_output1 = self.dim_output
        self.dim_output2 = self.dim_output

        self.hard_weight = Arg.hard_weight
        self.cl_weight = Arg.cl_weight
        self.ot_weight = Arg.ot_weight

        self.ot = OT(self.device)

        if self.datatype == 'SPOTS':
           self.epochs = 400
           self.weight_factors = [Arg.RNA_weight, Arg.ADT_weight, 1, 1] # [1,5,1,1]

        elif self.datatype == 'Stereo-CITE-seq':
           self.epochs = 800
           self.weight_factors = [Arg.RNA_weight, Arg.ADT_weight, 1, 5] # [1,10,1,10]

        elif self.datatype == '10x':
           self.epochs = 200
           self.weight_factors = [Arg.RNA_weight, Arg.ADT_weight, 1, 1] # [1,5,1,10]

        elif self.datatype == 'Spatial-epigenome-transcriptome':
           self.epochs = 600
           self.weight_factors = [Arg.RNA_weight, Arg.ADT_weight, 1, 1] # [1,10,1,1]

        elif self.datatype == 'MISAR':
           self.epochs = 600
           self.weight_factors = [Arg.RNA_weight, Arg.ADT_weight, 1, 1]

        elif self.datatype == 'DBit':
           self.epochs = 1000
           self.weight_factors = [Arg.RNA_weight, Arg.ADT_weight, 1, 1] # [1,5,1,1]

    def train(self):
        self.model = Encoder_overall(self.dim_input1, self.dim_output1, self.dim_input2, self.dim_output2).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate,
                                          weight_decay=self.weight_decay)

        self.model.train()

        for epoch in tqdm(range(self.epochs)):
            if epoch == 0:
                mask = neg_mask = None
            elif epoch == 1:
                self.adj_spatial_omics1, self.adj_spatial_omics2, pos_mask, neg_mask, mask = self.construct_graph_by_spatial(results['emb_latent_omics1'].clone().detach(), results['emb_latent_omics2'].clone().detach())
            else:
                neg_mask_bak = neg_mask
                self.adj_spatial_omics1, self.adj_spatial_omics2, pos_mask, neg_mask, mask = self.construct_graph_by_spatial(results['emb_latent_omics1'].clone().detach(), results['emb_latent_omics2'].clone().detach())
                neg_mask = (neg_mask_bak | neg_mask)

            self.model.train()
            results = self.model(self.features_omics1, self.features_omics2, self.adj_spatial_omics1, self.adj_feature_omics1, self.adj_spatial_omics2, self.adj_feature_omics2, epoch)

            # reconstruction loss
            loss_recon_omics1 =  F.mse_loss(self.features_omics1, results['emb_recon_spa_omics1'])
            loss_recon_omics2 =  F.mse_loss(self.features_omics2, results['emb_recon_spa_omics2'])

            # correspondence loss (cross reconstruction)
            loss_corr_omics1 = F.mse_loss(results['emb_latent_omics1'], results['emb_latent_omics1_across_recon'])
            loss_corr_omics2 = F.mse_loss(results['emb_latent_omics2'], results['emb_latent_omics2_across_recon'])

            loss_con = self.weight_factors[0]*loss_recon_omics1 + self.weight_factors[1]*loss_recon_omics2 + self.weight_factors[2]*loss_corr_omics1 + self.weight_factors[3]*loss_corr_omics2

            loss_ot = self.ot_loss(results['emb_latent_feature_omics1'], results['emb_latent_spatial_omics1']) + self.ot_loss(results['emb_latent_feature_omics2'], results['emb_latent_spatial_omics2'])

            loss_cl = (self.single_view_cl_Loss(results['emb_latent_combined'], neg_mask, mask, epoch, weight=self.hard_weight))

            loss = loss_con + loss_ot * self.ot_weight + loss_cl * self.cl_weight

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print("Model training finished!\n")

        with torch.no_grad():
            self.model.eval()
            results = self.model(self.features_omics1, self.features_omics2, self.adj_spatial_omics1, self.adj_feature_omics1, self.adj_spatial_omics2, self.adj_feature_omics2, epoch)

        # L2 normalization
        emb_omics1 = F.normalize(results['emb_latent_omics1'], p=2, eps=1e-12, dim=1)
        emb_omics2 = F.normalize(results['emb_latent_omics2'], p=2, eps=1e-12, dim=1)
        emb_combined = F.normalize(results['emb_latent_combined'], p=2, eps=1e-12, dim=1)

        output = {
                  'emb_latent_omics1': emb_omics1.detach().cpu().numpy(),
                  'emb_latent_omics2': emb_omics2.detach().cpu().numpy(),
                  'EMERGE': emb_combined.detach().cpu().numpy(),
                  'alpha': results['alpha'].detach().cpu().numpy()
                  }
        return output

    def single_view_cl_Loss(self, view1, neg_mask, mask, epoch, temperature=0.25, weight=200):
        if epoch >0:
            view1 = F.normalize(view1, dim=1)
            ttl_score = torch.matmul(view1, view1.transpose(0, 1))
            most_neg_score = (torch.exp(ttl_score * neg_mask/ temperature)).sum(dim=1) #
            neg_score = (torch.exp(ttl_score * (~mask)/ temperature)).sum(dim=1) #
            cl_loss = - torch.log(1/ (neg_score + most_neg_score * (weight) + 10e-10)) #
            return torch.mean(cl_loss)
        else:
            return(0)

    def ot_loss(self, pos_inter, source_inter):
        batch_size = 512  # adjust to available GPU memory
        device = source_inter.device

        # random shuffling
        idx = torch.randperm(source_inter.shape[0], device=device)
        shuffled_source = source_inter[idx]
        shuffled_target = pos_inter[idx]  # aligned samples

        total_loss = 0.0
        num_batches = 0

        for i in range(0, source_inter.shape[0], batch_size):
            # current batch
            batch_source = shuffled_source[i:i+batch_size]
            batch_target = shuffled_target[i:i+batch_size]

            # skip empty batch
            if batch_source.shape[0] == 0:
                continue

            # OT loss of the current batch
            batch_loss = self.ot(batch_source, batch_target)
            total_loss += batch_loss * batch_source.shape[0]  # weighted sum
            num_batches += batch_source.shape[0]

        # overall mean loss
        return total_loss / num_batches

    def compute_normalized_laplacian(self, edge_index, edge_weight, num_nodes, normalization='sym'):
        from torch_scatter import scatter_add
        row, col = edge_index[0].long(), edge_index[1].long()
        deg = scatter_add(edge_weight.float(), row, dim=0, dim_size=num_nodes)

        if normalization == 'sym':
            deg_inv_sqrt = deg.pow_(-0.5)
            deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
            edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        elif normalization == 'rw':
            deg_inv = 1.0 / deg
            deg_inv.masked_fill_(deg_inv == float('inf'), 0)
            edge_weight = deg_inv[row] * edge_weight
        size = torch.Size((num_nodes, num_nodes))
        return torch.sparse.FloatTensor(edge_index, edge_weight.float(), size)


    def construct_graph_by_spatial(self, emb_1, emb_2, top=0.45):
        context_norm_1 = emb_1.div(torch.norm(emb_1, p=2, dim=-1, keepdim=True))
        sim_1 = torch.mm(context_norm_1, context_norm_1.transpose(1, 0))
        context_norm_2 = emb_2.div(torch.norm(emb_2, p=2, dim=-1, keepdim=True))
        sim_2 = torch.mm(context_norm_2, context_norm_2.transpose(1, 0))

        sim_1_mean = torch.quantile(sim_1, top, dim=1, keepdim=True)
        sim_2_mean = torch.quantile(sim_2, top, dim=1, keepdim=True)

        mask_1 = sim_1 > sim_1_mean # keep pairs above the quantile
        mask_2 = sim_2 > sim_2_mean #
        mask = (mask_1 & mask_2) # keep pairs passing both modalities

        n_samples = sim_1.shape[0]
        adj_spatial_omics, pos_mask, neg_mask = self.prune_sparse_graph_by_mask(mask=mask, sparse_graph=self.adj_spatial_omics1, n_samples=n_samples)

        print('Spatial Graph Edges' + str(adj_spatial_omics._indices().shape[1]))
        return adj_spatial_omics.to(self.device), adj_spatial_omics.to(self.device), pos_mask, neg_mask, mask

    def prune_sparse_graph_by_mask(self, mask, sparse_graph, n_samples):
        dense_graph_1 = sparse_graph.coalesce().to_dense()
        # drop edges failing the mask (using the latent embedding directly performed worse)
        sparse_matrix_1 = (dense_graph_1 * mask).to_sparse()
        clipped_values_1 = torch.ones(sparse_matrix_1._indices().shape[1]).to(self.device)
        clipped_matrix_1 = torch.sparse_coo_tensor(sparse_matrix_1._indices(), clipped_values_1, sparse_matrix_1.size())
        adj_spatial_omics = self.compute_normalized_laplacian(clipped_matrix_1._indices(),  clipped_matrix_1._values(), n_samples, normalization='sym')

        # masked-out edges become negative pairs
        sparse_matrix = sparse_graph.coalesce()
        indices = sparse_matrix._indices()  # non-zero indices (2 x nnz)
        rows, cols = sparse_matrix.size()
        neg_mask = torch.zeros(rows, cols, dtype=torch.bool).to(self.device)
        # positions masked out of the spatial graph
        neg_indices = indices[:, ~mask[indices[0], indices[1]]]
        # positions whose mask is False become negatives
        neg_mask[neg_indices[0], neg_indices[1]] = True

        pos_mask = torch.zeros(rows, cols, dtype=torch.bool).to(self.device)
        pos_indices = indices[:, mask[indices[0], indices[1]]]
        pos_mask[pos_indices[0], pos_indices[1]] = True

        return adj_spatial_omics, pos_mask, neg_mask
