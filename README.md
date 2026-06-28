# EEG Preprocessing and Graph Construction

This module provides the pipeline to transform raw EEG data into segmented epochs and generate multi-view topological adjacency matrices required for HFG-Net.

## Workflow Overview
The preprocessing pipeline consists of three main stages:
1. **Signal Preprocessing:** Band-pass filtering (0.5–64 Hz), downsampling to 256 Hz, and channel selection (19 channels).
2. **Segmentation:** Sliding window segmentation with a 4-second window and 1-second overlap.
3. **Graph Construction:** - Calculation of full-band Amplitude Coupling (PC) and Phase Synchronization (PLV) matrices.
   - High-frequency guided sparsification using KNN and PT to generate topological priors.

## Directory Structure
- `preprocess_eeg.m`: Main script for data filtering, segmentation, and base matrix computation.
- `preprocess_high_freq_guided.m`: Sparsification script to generate KNN/PT masks based on high-frequency (Beta/Gamma) connectivity.

## Output Specifications
Each segment is saved as a `segment_*.mat` file containing the following variables:

### 1. Feature Data
* `segment_data`: EEG signal segment (Channels $\times$ Time, 19 $\times$ 1024).

### 2. Base Matrices (All-band)
* `seg_pc_matrix`: 19 $\times$ 19 Pearson Correlation matrix.
* `seg_plv_matrix`: 19 $\times$ 19 Phase Locking Value matrix.

### 3. Topological Priors (High-frequency Guided)
These masks are computed exclusively from the 13–45 Hz high-frequency band to serve as structural priors:
* **PC-based Priors:**
  - `pc_knn_mask`: Binary mask generated via K-Nearest Neighbors ($K=5$).
  - `pc_pt_mask`: Binary mask generated via Proportional Thresholding ($P=25\%$).
* **PLV-based Priors:**
  - `plv_knn_mask`: Binary mask generated via K-Nearest Neighbors ($K=5$).
  - `plv_pt_mask`: Binary mask generated via Proportional Thresholding ($P=25\%$).

## Usage
1. Configure the `ROOT` path in both MATLAB scripts.
2. Run `preprocess_eeg.m` to generate base matrices.
3. Run `preprocess_high_freq_guided.m` to generate the KNN/PT masks.
4. Ensure the output structure matches the directory expectations of the Python data loader.

> **Note:** To maintain strict subject-level isolation, graph construction (including high-frequency filtering and sparsification) is performed dynamically on a per-sample basis within each independent epoch.
