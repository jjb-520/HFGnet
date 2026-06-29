HFG-Net: High-Frequency Guided Dual-Branch Neural Network for EEG Analysis

This repository contains the official implementation of HFG-Net, a novel dual-branch deep learning framework (CNN + GCN) designed for EEG-based classification tasks (e.g., ASD vs. Typical Development).

The model leverages High-Frequency Guided Topological Priors (using KNN and PT sparsification) and dynamic sample-level feature fusion to capture both multi-scale temporal dynamics and multi-view spatial topologies from EEG signals.

✨ Key Features

Multi-View Graph Convolutional Network (GCN): Processes Amplitude Coupling (PC) and Phase Synchronization (PLV) functional connectivity matrices.

High-Frequency Guidance: Integrates topological priors (K-Nearest Neighbors & Proportional Thresholding masks) derived exclusively from high-frequency bands (Beta/Gamma) to guide full-band network learning.

Multi-Scale Temporal CNN: Extracts temporal features across different downsampling scales, enhanced by a Combined Temporal-Channel Attention Module (CTAM).

Dynamic Feature Fusion: Utilizes a sample-level gating network to adaptively assign importance weights to the CNN and GCN branches for each individual subject.

Clinical Deployment Simulation: Includes a built-in measure_single_sample_latency function to evaluate real-time inference speed (Batch Size = 1).

📂 Repository Structure

The repository is divided into MATLAB preprocessing scripts and Python deep learning scripts:

Preprocessing (MATLAB)

preprocess_eeg.m: Main script for data filtering, segmentation, and base matrix computation.

preprocess_high_freq_guided.m: Sparsification script to generate KNN/PT masks based on high-frequency connectivity.

Deep Learning (Python)

model.py: Defines the complete architecture of HFG-Net.

trainer.py: A robust, custom training engine supporting class-weight balancing, early stopping, dynamic learning rate scheduling (ReduceLROnPlateau), and metric tracking.

main.py: The main execution script orchestrating the 5-fold cross-validation pipeline, data loading, inference, and visualization generation.

🛠️ Requirements

For Preprocessing:

MATLAB (R2019b or newer recommended)

Signal Processing Toolbox

For Deep Learning:

Python 3.8+

PyTorch >= 1.10.0 (CUDA highly recommended)

NumPy, scikit-learn, Matplotlib, Seaborn

Install the Python dependencies using:

pip install torch numpy scikit-learn matplotlib seaborn


🚀 Pipeline Stage 1: EEG Preprocessing and Graph Construction (MATLAB)

This module provides the pipeline to transform raw EEG data into segmented epochs and generate multi-view topological adjacency matrices required for HFG-Net.

Workflow Overview

Signal Preprocessing: Band-pass filtering (0.5–64 Hz), downsampling to 256 Hz, and channel selection (19 channels).

Segmentation: Sliding window segmentation with a 4-second window and 1-second overlap.

Graph Construction: - Calculation of full-band Pearson Correlation (PC) and Phase Locking Value (PLV) matrices.

High-frequency guided sparsification using KNN and PT to generate topological priors.

Output Specifications

Each segment is saved as a segment_*.mat file containing the following variables:

Feature Data: segment_data (EEG signal segment, 19 $\times$ 1024).

Base Matrices (All-band): seg_pc_matrix, seg_plv_matrix (19 $\times$ 19).

Topological Priors (High-frequency Guided): Computed exclusively from the 13–45 Hz high-frequency band.

PC-based: pc_knn_mask ($K=5$), pc_pt_mask ($P=25\%$).

PLV-based: plv_knn_mask ($K=5$), plv_pt_mask ($P=25\%$).

Usage (MATLAB)

Configure the ROOT path in both MATLAB scripts.

Run preprocess_eeg.m to generate base matrices.

Run preprocess_high_freq_guided.m to generate the KNN/PT masks.

Note: To maintain strict subject-level isolation, graph construction (including high-frequency filtering and sparsification) is performed dynamically on a per-sample basis within each independent epoch.

🚀 Pipeline Stage 2: Model Training and Evaluation (Python)

Dataset Preparation

The Python data loader expects the aggregated .mat outputs to be saved as .npy dictionaries for efficient loading. Each dictionary must contain:

X: Raw EEG signals. Shape: (Num_Samples, 19 * 1024)

y: Labels (0 or 1). Shape: (Num_Samples,)

adj1 to adj6: 6 Topological adjacency matrices (Full-band matrices + KNN/PT masks for both PLV and PC). Shape for each: (Num_Samples, 19, 19)

Usage (Python)

To start the 5-fold cross-validation training and testing pipeline, simply run:

python main.py


Configuration: You can modify the hyper-parameters directly in the main() function inside main.py (e.g., data_root, n_channels=19, seq_length=1024, batch_size=64, learning_rate=1e-3).

📈 Output Artifacts

Upon completion, the script automatically generates a model_cnn_gcn_fusion/ directory containing:

Model Weights: best_model_fold_X.pt for each fold.

Performance Results: fusion_model_results.npz containing all metrics, predictions, and scaler objects.

Visualizations:

cv_training_history.png: Loss and accuracy curves.

test_vote_cm.png: Confusion matrix of the final ensemble voting.

fusion_weights/: Directory containing dynamic fusion weight distributions and t-SNE manifold visualizations for CNN, GCN, and Fused features.

Latency Report: Standard output will print the average real-time single-sample inference latency in milliseconds.
