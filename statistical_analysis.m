%% HFG-Net: Group-level Statistical Analysis (ASD vs TD)
% Performs t-test with Benjamini-Hochberg FDR correction.
% Exports significant edges for brain network visualization.

clear; clc; close all;

%% 1. Configuration
% Set paths using fullfile for cross-platform compatibility
ROOT = 'D:\Downloads\eeglab_current\128';
cfg.asd_root = fullfile(ROOT, 'asd', 'delta');
cfg.td_root  = fullfile(ROOT, 'td', 'delta');
cfg.fdr_q    = 0.05;
cfg.num_subjects = 50;

%% 2. Data Loading
fprintf('Loading data...\n');
[asd_pc, asd_plv] = load_group_data(cfg.asd_root, cfg.num_subjects, 'asd');
[td_pc, td_plv]   = load_group_data(cfg.td_root, cfg.num_subjects, 'td');

n_subjects = cfg.num_subjects;
df = n_subjects * 2 - 2;

%% 3. Statistical Testing (T-test)
[t_pc,  p_pc]  = compute_ttest(asd_pc, td_pc, n_subjects);
[t_plv, p_plv] = compute_ttest(asd_plv, td_plv, n_subjects);

%% 4. FDR Correction (Benjamini-Hochberg)
sig_pc  = apply_fdr(p_pc, cfg.fdr_q);
sig_plv = apply_fdr(p_plv, cfg.fdr_q);

%% 5. Export Results
% Masked t-statistics (Significant edges only)
export_results(t_pc .* sig_pc, 'fdr_pc.edge');
export_results(t_plv .* sig_plv, 'fdr_plv.edge');

save('group_statistics_fdr005.mat', 't_pc', 't_plv', 'sig_pc', 'sig_plv');
fprintf('Analysis complete. Results exported to .edge and .mat files.\n');

%% Helper Functions
function [t, p] = compute_ttest(asd, td, n)
    mu1 = mean(asd, 3); mu2 = mean(td, 3);
    var1 = var(asd, 0, 3); var2 = var(td, 0, 3);
    t = (mu1 - mu2) ./ sqrt(var1/n + var2/n);
    p = 2 * (1 - tcdf(abs(t), 2*n-2));
end

function sig = apply_fdr(p_mat, q)
    mask = tril(true(size(p_mat,1)), -1);
    p_vec = p_mat(mask);
    [sorted_p, idx] = sort(p_vec);
    crit = (1:length(p_vec))' / length(p_vec) * q;
    k = find(sorted_p <= crit, 1, 'last');
    h = false(size(p_vec));
    if ~isempty(k), h(idx(1:k)) = true; end
    res = zeros(size(p_mat));
    res(mask) = h;
    sig = res + res';
end

function export_results(mat, filename)
    mat(isnan(mat)) = 0;
    dlmwrite(filename, abs(mat), 'delimiter', '\t', 'precision', 15);
end

function [pc, plv] = load_group_data(root, n, group_label)
    pc = []; plv = []; 
    for i = 1:n
        % Flexible naming: attempts to load 'group_i_full.mat'
        f_name = fullfile(root, sprintf('%s_%d_full.mat', group_label, i));
        if exist(f_name, 'file')
            data = load(f_name);
            pc = cat(3, pc, data.full_pc_matrix);
            plv = cat(3, plv, data.full_plv_matrix);
        else
            warning('File not found: %s', f_name);
        end
    end
end
