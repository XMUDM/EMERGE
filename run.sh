#!/bin/bash

############ MISAR_seq_mouse_E15_brain

python -u main.py --file_fold './Data/MISAR_seq_mouse_E15_brain/' --data_type 'MISAR' --n_clusters 12 --KNN_k 20 --RNA_weight 5 --ADT_weight 5 --GT_path './Data/MISAR_seq_mouse_E15_brain/GT_labels.txt' --vis_out_path 'results/EMERGE_MISAR_E15.png' --txt_out_path 'results/EMERGE_MISAR_E15.txt' --save_metrics_path './results/EMERGE_MISAR_E15_metrics.txt' --save_path './results/EMERGE_MISAR_E15.h5ad' --hard_weight 50 --cl_weight 10 --ot_weight 0.1

############ Stereo-CITE-seq Mouse_Thymus

python -u main.py --file_fold './Data/Mouse_Thymus' --data_type 'Stereo-CITE-seq' --n_clusters 8 --KNN_k 20 --RNA_weight 5 --ADT_weight 10 --vis_out_path 'results/EMERGE_Mouse_Thymus.png' --txt_out_path 'results/EMERGE_Mouse_Thymus.txt' --save_metrics_path './results/EMERGE_Mouse_Thymus_metrics.txt' --save_path './results/EMERGE_Mouse_Thymus.h5ad' --hard_weight 100 --cl_weight 5 --ot_weight 0

############ Spatial-epigenome-transcriptome Dataset10 (H3K27me3)

python -u main.py --file_fold './Data/Dataset10_Mouse_Brain_H3K27me3' --data_type 'Spatial-epigenome-transcriptome' --n_clusters 18 --KNN_k 20 --RNA_weight 5 --ADT_weight 5 --vis_out_path 'results/EMERGE_MB_H3K27me3.png' --txt_out_path 'results/EMERGE_MB_H3K27me3.txt' --save_metrics_path './results/EMERGE_MB_H3K27me3_metrics.txt' --save_path './results/EMERGE_MB_H3K27me3.h5ad' --hard_weight 800 --cl_weight 1 --ot_weight 0.005

############ Visium-MALDI parkinsonian Mouse_Brain (3 serial sections)
# NOTE: --n_top_genes 1000 overrides the main.py default (3000); see results/mousebrain/final/MANIFEST.md

python -u main.py --file_fold './Data/Visium_MALDI_Mouse_Brain_SectionA' --data_type 'RNA-METABOLITE' --n_clusters 11 --n_top_genes 1000 --meta_hvg 1000 --KNN_k 20 --RNA_weight 1 --ADT_weight 1 --hard_weight 500 --cl_weight 1 --ot_weight 0.1 --epochs 600 --rna_pcs 50 --meta_pcs 50 --spatial_knn 6 --seed 1 --GT_path './Data/Visium_MALDI_Mouse_Brain_SectionA/GT_labels.txt' --vis_out_path 'results/mousebrain/EMERGE_MouseBrain_SectionA.png' --txt_out_path 'results/mousebrain/EMERGE_MouseBrain_SectionA.txt' --save_metrics_path 'results/mousebrain/EMERGE_MouseBrain_SectionA_metrics.txt' --save_path 'results/mousebrain/EMERGE_MouseBrain_SectionA.h5ad'

python -u main.py --file_fold './Data/Visium_MALDI_Mouse_Brain_SectionB' --data_type 'RNA-METABOLITE' --n_clusters 11 --n_top_genes 1000 --meta_hvg 1000 --KNN_k 5 --RNA_weight 5 --ADT_weight 5 --hard_weight 500 --cl_weight 1 --ot_weight 0.1 --epochs 600 --rna_pcs 50 --meta_pcs 50 --spatial_knn 6 --seed 2024 --GT_path './Data/Visium_MALDI_Mouse_Brain_SectionB/GT_labels.txt' --vis_out_path 'results/mousebrain/EMERGE_MouseBrain_SectionB.png' --txt_out_path 'results/mousebrain/EMERGE_MouseBrain_SectionB.txt' --save_metrics_path 'results/mousebrain/EMERGE_MouseBrain_SectionB_metrics.txt' --save_path 'results/mousebrain/EMERGE_MouseBrain_SectionB.h5ad'

python -u main.py --file_fold './Data/Visium_MALDI_Mouse_Brain_SectionC' --data_type 'RNA-METABOLITE' --n_clusters 11 --n_top_genes 1000 --meta_hvg 1000 --KNN_k 10 --RNA_weight 5 --ADT_weight 5 --hard_weight 500 --cl_weight 1 --ot_weight 0.1 --epochs 600 --rna_pcs 50 --meta_pcs 50 --spatial_knn 6 --seed 2024 --GT_path './Data/Visium_MALDI_Mouse_Brain_SectionC/GT_labels.txt' --vis_out_path 'results/mousebrain/EMERGE_MouseBrain_SectionC.png' --txt_out_path 'results/mousebrain/EMERGE_MouseBrain_SectionC.txt' --save_metrics_path 'results/mousebrain/EMERGE_MouseBrain_SectionC_metrics.txt' --save_path 'results/mousebrain/EMERGE_MouseBrain_SectionC.h5ad'
