#!/bin/bash
# EMERGE — main experiments on all datasets
# One canonical configuration per dataset.

############ MISAR_seq_mouse_E15_brain

python -u main.py --file_fold './Data/MISAR_seq_mouse_E15_brain/' --data_type 'MISAR' --n_clusters 12 --KNN_k 20 --RNA_weight 5 --ADT_weight 5 --GT_path './Data/MISAR_seq_mouse_E15_brain/GT_labels.txt' --vis_out_path 'results/EMERGE_MISAR_E15.png' --txt_out_path 'results/EMERGE_MISAR_E15.txt' --save_metrics_path './results/EMERGE_MISAR_E15_metrics.txt' --save_path './results/EMERGE_MISAR_E15.h5ad' --hard_weight 50 --cl_weight 10 --ot_weight 0.1

############ Stereo-CITE-seq Mouse_Thymus

python -u main.py --file_fold './Data/Mouse_Thymus' --data_type 'Stereo-CITE-seq' --n_clusters 8 --KNN_k 20 --RNA_weight 5 --ADT_weight 10 --vis_out_path 'results/EMERGE_Mouse_Thymus.png' --txt_out_path 'results/EMERGE_Mouse_Thymus.txt' --save_metrics_path './results/EMERGE_Mouse_Thymus_metrics.txt' --save_path './results/EMERGE_Mouse_Thymus.h5ad' --hard_weight 100 --cl_weight 5 --ot_weight 0

############ Spatial-epigenome-transcriptome Dataset10 (H3K27me3)

python -u main.py --file_fold './Data/Dataset10_Mouse_Brain_H3K27me3' --data_type 'Spatial-epigenome-transcriptome' --n_clusters 18 --KNN_k 20 --RNA_weight 5 --ADT_weight 5 --vis_out_path 'results/EMERGE_MB_H3K27me3.png' --txt_out_path 'results/EMERGE_MB_H3K27me3.txt' --save_metrics_path './results/EMERGE_MB_H3K27me3_metrics.txt' --save_path './results/EMERGE_MB_H3K27me3.h5ad' --hard_weight 800 --cl_weight 1 --gpu_id 2 --ot_weight 0.05 
