%% HFG-Net: High-Frequency Guided KNN/PT Sparsification Script
% Description: This script performs sparsification on high-frequency EEG 
%              connectivity matrices to generate topological priors.
%              These priors guide the processing of all-band PC and PLV matrices.
% Author: [Your Name/Anonymized]

clear; clc; close all;

%% 1. Configuration of Data Paths
% Adjust these paths to your project structure
ROOT = 'D:\Downloads\eeglab_current\64'; 
cfg.asd_src = fullfile(ROOT,  'asd', 'high_freq');
cfg.td_src  = fullfile(ROOT, 'td', 'high_freq');
cfg.asd_target = fullfile(ROOT,  'high_freq', 'ASDKNNPT');
cfg.td_target  = fullfile(ROOT,  'high_freq', 'TDKNNPT');

%% 2. Execution
process_group(cfg.asd_src, cfg.asd_target, 'asd');
process_group(cfg.td_src, cfg.td_target, 'td');

%% 3. Processing Functions
function process_group(src_root, target_root, group_name)
    if ~exist(target_root, 'dir'), mkdir(target_root); end
    
    subj_folders = dir(fullfile(src_root, [group_name '_*']));
    subj_folders = {subj_folders.name};
    
    for s = 1:length(subj_folders)
        src_path = fullfile(src_root, subj_folders{s});
        target_path = fullfile(target_root, subj_folders{s});
        if ~exist(target_path, 'dir'), mkdir(target_path); end
        
        fprintf('Processing subject: %s\n', subj_folders{s});
        
        segment_files = dir(fullfile(src_path, 'segment_*.mat'));
        for seg = 1:length(segment_files)
            load(fullfile(src_path, segment_files(seg).name));
            
            % Generate KNN and PT masks based on high-frequency topology
            % Note: seg_pc_high and seg_plv_high must be computed 
            % from the 13-45 Hz filtered signals before this step.
            pc_knn_mask = knn_threshold(seg_pc_high, 5);
            pc_pt_mask  = pt_threshold(seg_pc_high, 25);
            
            plv_knn_mask = knn_threshold(seg_plv_high, 5);
            plv_pt_mask  = pt_threshold(seg_plv_high, 25);
            
            % Save all variables to the target directory
            save(fullfile(target_path, segment_files(seg).name), ...
                'segment_data', 'segment_info', ...
                'seg_pc_matrix', 'seg_plv_matrix', ...
                'pc_knn_mask', 'pc_pt_mask', ...
                'plv_knn_mask', 'plv_pt_mask');
        end
    end
end

%% KNN Sparsification (Keep top K strong connections)
function A_knn = knn_threshold(A, k)
    n = size(A, 1);
    A_knn = zeros(size(A));
    A_temp = A .* (1 - eye(n)); 
    for i = 1:n
        [~, top_idx] = sort(A_temp(i, :), 'descend');
        A_knn(i, top_idx(1:k)) = 1; 
    end
    % Ensure symmetry
    A_knn = (A_knn + A_knn') > 0;
end

%% PT Sparsification (Keep top P% strong connections)
function A_pt = pt_threshold(A, p)
    n = size(A, 1);
    A_temp = A .* (1 - eye(n));
    non_zero_vals = A_temp(A_temp > 0);
    if isempty(non_zero_vals), A_pt = zeros(n); return; end
    
    threshold = quantile(non_zero_vals, 1 - p/100);
    A_pt = (A_temp >= threshold);
    % Ensure symmetry
    A_pt = (A_pt + A_pt') > 0;
end