%% HFG-Net: EEG Preprocessing and Graph Construction Script
% This script performs band-pass filtering, segmentation, and 
% computes PC and PLV adjacency matrices for HFG-Net.

clear; clc; close all;

%% 1. Path and Parameters Configuration
% Use relative paths based on your project root
ROOT = 'D:\Downloads\eeglab_current\128'; 
cfg.input_folder = fullfile(ROOT, 'td');
cfg.output_base_folder = fullfile(ROOT, 'data', 'td19');

if ~exist(cfg.output_base_folder, 'dir'), mkdir(cfg.output_base_folder); end

% Channel Selection (19 channels)
cfg.selected_channels = [1,2,4,5,7,8,10,11,12,15,19,21,22,23,24,27,28,29,30];

% Preprocessing parameters
cfg.filter_order = 4;
cfg.lowpass_freq = 64;
cfg.target_srate = 256;

% Segmentation parameters
cfg.segment_length = 4; % seconds
cfg.overlap_length = 1; % seconds

%% 2. Main Processing Loop
file_list = dir(fullfile(cfg.input_folder, '*.mat'));

for i = 1:length(file_list)
    fprintf('Processing file %d/%d: %s\n', i, length(file_list), file_list(i).name);
    
    % Load Data
    mat_data = load(fullfile(cfg.input_folder, file_list(i).name));
    
    % Extract EEG field (standardizing variable name)
    if isfield(mat_data, 'data_end'), eeg_data = mat_data.data_end;
    else, error('EEG field not found.'); end
    
    % Get Sampling Rate
    srate = 1000; if isfield(mat_data, 'srate'), srate = mat_data.srate; end
    
    % Transpose if necessary (ensure Rows=Channels)
    [r, c] = size(eeg_data);
    if r > c, eeg_data = eeg_data'; end
    eeg_data = eeg_data(cfg.selected_channels, :);
    
    % Filtering & Downsampling
    [b, a] = butter(cfg.filter_order, cfg.target_srate / (srate/2), 'low');
    eeg_down = resample(filtfilt(b, a, eeg_data')', cfg.target_srate, srate);
    [b, a] = butter(cfg.filter_order, cfg.lowpass_freq / (cfg.target_srate/2), 'low');
    eeg_filt = filtfilt(b, a, eeg_down')';
    
    % Segmentation
    step = (cfg.segment_length - cfg.overlap_length) * cfg.target_srate;
    len = cfg.segment_length * cfg.target_srate;
    num_segments = floor((size(eeg_filt, 2) - cfg.overlap_length*cfg.target_srate) / step);
    
    output_folder = fullfile(cfg.output_base_folder, sprintf('td_%d', i));
    if ~exist(output_folder, 'dir'), mkdir(output_folder); end
    
    for seg = 1:num_segments
        start_idx = (seg - 1) * step + 1;
        segment_data = eeg_filt(:, start_idx : start_idx + len - 1);
        
        % Calculate Adjacency Matrices
        % PC Matrix
        seg_pc_matrix = abs(corrcoef(segment_data'));
        
        % PLV Matrix
        analytic_signal = hilbert(segment_data' - mean(segment_data, 2)');
        phases = angle(analytic_signal);
        n_ch = size(segment_data, 1);
        seg_plv_matrix = zeros(n_ch, n_ch);
        for ch1 = 1:n_ch
            for ch2 = 1:n_ch
                phase_diff = phases(:, ch1) - phases(:, ch2);
                seg_plv_matrix(ch1, ch2) = abs(mean(exp(1i * phase_diff)));
            end
        end
        
        % Save segment
        save(fullfile(output_folder, sprintf('segment_%03d.mat', seg)), ...
            'segment_data', 'seg_pc_matrix', 'seg_plv_matrix');
    end
end
fprintf('All files processed successfully.\n');