import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# Import the updated 19-channel, 6-matrix model
from model import CombinedCNN_GCN
from trainer import Trainer


# -------------------------- 1. Utility Functions --------------------------

def measure_single_sample_latency(model, test_time_X, test_gcn_X, test_adj, device, num_runs=100):
    """
    Measures single-sample inference latency to simulate real-time clinical deployment (Batch Size = 1).
    """
    model.eval()

    # Extract the first sample to force Batch Size = 1
    single_time_X = test_time_X[0:1].to(device)
    single_gcn_X = test_gcn_X[0:1].to(device)
    single_adj = test_adj[0:1].to(device)

    # Unbind the 6 adjacency matrices (PLV: 1-3, PC: 4-6)
    a1, a2, a3, a4, a5, a6 = torch.unbind(single_adj, dim=1)

    print("\n--- Evaluating Single-Sample Inference Latency ---")

    # 1. GPU Warm-up
    with torch.no_grad():
        for _ in range(20):
            _ = model(single_time_X, single_gcn_X, a1, a2, a3, a4, a5, a6)

    # 2. Formal Measurement
    if device.type == 'cuda':
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start_event.record()

        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(single_time_X, single_gcn_X, a1, a2, a3, a4, a5, a6)

        end_event.record()
        torch.cuda.synchronize()

        total_time_ms = start_event.elapsed_time(end_event)
    else:
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(single_time_X, single_gcn_X, a1, a2, a3, a4, a5, a6)
        total_time_ms = (time.time() - start_time) * 1000

    avg_latency_ms = total_time_ms / num_runs
    print(f"Total time for {num_runs} runs: {total_time_ms:.2f} ms")
    print(f"🌟 True Single-Sample Latency: {avg_latency_ms:.2f} ms")

    return avg_latency_ms


# -------------------------- 2. Visualization Functions --------------------------

def visualize_tsne(features, labels, save_path, class_names=['Typical', 'ASD']):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    features_tsne = tsne.fit_transform(features_scaled)

    plt.figure(figsize=(8, 6))
    mask_asd = labels == 1
    mask_ctl = labels == 0
    plt.scatter(features_tsne[mask_asd, 0], features_tsne[mask_asd, 1], c='purple', label='ASD', s=50, alpha=0.8)
    plt.scatter(features_tsne[mask_ctl, 0], features_tsne[mask_ctl, 1], c='yellow', label='Control', s=50, alpha=0.8)
    plt.legend()
    plt.title('t-SNE Visualization of Model Features')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"t-SNE visualization saved to: {save_path}")


def visualize_fusion_weights(weights, save_path, class_names=['Typical', 'ASD'], true_labels=None):
    cnn_weights = weights[:, 0]
    gcn_weights = weights[:, 1]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(cnn_weights, bins=20, alpha=0.5, label='CNN Weights', color='blue')
    plt.hist(gcn_weights, bins=20, alpha=0.5, label='GCN Weights', color='orange')
    plt.axvline(x=0.5, color='red', linestyle='--', label='Balance Threshold')
    plt.xlabel('Weight Value')
    plt.ylabel('Number of Samples')
    plt.title('Distribution of Dynamic Fusion Weights')
    plt.legend()

    if true_labels is not None:
        plt.subplot(1, 2, 2)
        data = [cnn_weights[true_labels == i] for i in range(len(class_names))] + \
               [gcn_weights[true_labels == i] for i in range(len(class_names))]

        bp = plt.boxplot(data, tick_labels=class_names * 2)
        plt.xticks(rotation=45)
        plt.ylabel('Weight Value')
        plt.title('Weight Distribution by Class')
        plt.scatter([], [], c='blue', label='CNN Weights')
        plt.scatter([], [], c='orange', label='GCN Weights')
        plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Weight visualization saved to: {save_path}")


def visualize_training_history(fold_history, save_path='training_history.png'):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    for fold_idx, history in enumerate(fold_history):
        plt.plot(history['train_loss'], label=f'Fold {fold_idx + 1} Train')
        plt.plot(history['val_loss'], label=f'Fold {fold_idx + 1} Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    for fold_idx, history in enumerate(fold_history):
        plt.plot(history['train_acc'], label=f'Fold {fold_idx + 1} Train')
        plt.plot(history['val_acc'], label=f'Fold {fold_idx + 1} Val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training history saved to: {save_path}")


def visualize_confusion_matrix(cm, class_names=['Typical', 'ASD'], save_path='confusion_matrix.png'):
    plt.figure(figsize=(8, 6))
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_percent, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Percentage'})
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {save_path}")


# -------------------------- 3. Data Loading & Preprocessing --------------------------

def load_eeg_data(data_dirs, data_type='train', n_channels=21, seq_length=1024):
    time_X_list, gcn_X_list, y_list, adj_list = [], [], [], []

    for data_dir in data_dirs:
        if data_type == 'train':
            file_path = os.path.join(data_dir, 'train_data.npy')
        elif data_type == 'val':
            file_path = os.path.join(data_dir, 'val_data.npy')
        elif data_type == 'test':
            file_path = data_dir
        else:
            raise ValueError("data_type must be 'train', 'val', or 'test'")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        data = np.load(file_path, allow_pickle=True).item()

        # Load 6 matrices (PLV: 1-3, PC: 4-6)
        required_keys = ['X', 'y'] + [f'adj{i}' for i in range(1, 7)]
        missing_keys = [k for k in required_keys if k not in data]
        if missing_keys:
            raise ValueError(f"Missing keys in {file_path}: {missing_keys}")

        X = data['X']
        y = data['y']

        adjs = []
        for i in range(1, 7):
            adj = data[f'adj{i}']
            if adj.shape != (len(y), n_channels, n_channels):
                raise ValueError(
                    f"adj{i} shape error! Expected ({len(y)}, {n_channels}, {n_channels}), got {adj.shape}")
            adjs.append(adj)

        time_X = X.reshape(-1, n_channels, seq_length)
        gcn_X = X.reshape(-1, 1, n_channels, seq_length)
        adjs_stacked = np.stack(adjs, axis=1)

        time_X_list.append(time_X)
        gcn_X_list.append(gcn_X)
        y_list.append(y)
        adj_list.append(adjs_stacked)

        print(f"Loaded {data_type} data: {file_path}")

    time_X = np.vstack(time_X_list)
    gcn_X = np.vstack(gcn_X_list)
    y = np.concatenate(y_list)
    adj = np.vstack(adj_list)

    print(f"Total Data Shape -> time_X: {time_X.shape}, gcn_X: {gcn_X.shape}, y: {y.shape}, adj: {adj.shape}")
    return time_X, gcn_X, y, adj


def preprocess_data(time_X, gcn_X, y, adj, scaler=None):
    n_samples = len(y)
    assert time_X.shape[0] == gcn_X.shape[0] == adj.shape[0] == n_samples, "Sample size mismatch"

    if scaler is None:
        scaler = StandardScaler()
        time_X_reshaped = time_X.reshape(-1, time_X.shape[-1])
        time_X_processed = scaler.fit_transform(time_X_reshaped).reshape(time_X.shape)

        gcn_X_reshaped = gcn_X.reshape(-1, gcn_X.shape[-1])
        gcn_X_processed = scaler.transform(gcn_X_reshaped).reshape(gcn_X.shape)
    else:
        time_X_reshaped = time_X.reshape(-1, time_X.shape[-1])
        time_X_processed = scaler.transform(time_X_reshaped).reshape(time_X.shape)

        gcn_X_reshaped = gcn_X.reshape(-1, gcn_X.shape[-1])
        gcn_X_processed = scaler.transform(gcn_X_reshaped).reshape(gcn_X.shape)

    time_X_processed = torch.FloatTensor(time_X_processed)
    gcn_X_processed = torch.FloatTensor(gcn_X_processed)
    y_processed = torch.LongTensor(y)
    adj_processed = torch.FloatTensor(adj)

    return time_X_processed, gcn_X_processed, y_processed, adj_processed, scaler


# -------------------------- 4. Main Training & Evaluation Loop --------------------------

def main():
    data_root = "/root/autodl-tmp/5K_GCN_9adj"
    test_data_path = os.path.join(data_root, "test_data.npy")
    model_save_dir = "model_cnn_gcn_fusion"
    weights_dir = os.path.join(model_save_dir, "fusion_weights")

    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)

    # Hyperparameters
    n_channels = 21
    seq_length = 1024
    n_classes = 2
    input_size = (None, 1, n_channels, seq_length)

    epochs = 100
    batch_size = 64
    learning_rate = 1e-3
    patience = 15
    early_stopping_metric = 'val_loss'
    use_gpu = True
    random_state = 42

    GCN_hidden1 = 32
    GCN_hidden3 = 10
    dropout_rate = 0.4

    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    print(f"Using device: {device}")

    # -------------------------- Load Data --------------------------
    print("\n=== Loading Training Data ===")
    fold_dirs = [os.path.join(data_root, f'fold_{i + 1}') for i in range(5)]
    all_time_X, all_gcn_X, all_y, all_adj = load_eeg_data(
        fold_dirs, data_type='train', n_channels=n_channels, seq_length=seq_length
    )

    print("\n=== Loading Test Data ===")
    test_time_X, test_gcn_X, test_y, test_adj = load_eeg_data(
        [test_data_path], data_type='test', n_channels=n_channels, seq_length=seq_length
    )
    test_true_np = test_y

    raw_test_features = test_time_X.reshape(len(test_y), -1)
    visualize_tsne(raw_test_features, test_y, os.path.join(model_save_dir, 'raw_test_tsne.png'))

    # -------------------------- Cross-Validation --------------------------
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    fold_results, fold_history = [], []
    best_val_f1 = 0.0
    best_model_path = ""

    print(f"\n=== Starting 5-Fold Cross Validation ===")
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(all_time_X)):
        print(f"\n========== Fold {fold_idx + 1}/5 ==========")

        train_time_X, train_gcn_X = all_time_X[train_idx], all_gcn_X[train_idx]
        train_y, train_adj = all_y[train_idx], all_adj[train_idx]

        val_time_X, val_gcn_X = all_time_X[val_idx], all_gcn_X[val_idx]
        val_y, val_adj = all_y[val_idx], all_adj[val_idx]

        train_time_X, train_gcn_X, train_y, train_adj, scaler = preprocess_data(
            train_time_X, train_gcn_X, train_y, train_adj, scaler=None
        )
        val_time_X, val_gcn_X, val_y, val_adj, _ = preprocess_data(
            val_time_X, val_gcn_X, val_y, val_adj, scaler=scaler
        )

        train_time_X, train_gcn_X = train_time_X.to(device), train_gcn_X.to(device)
        train_adj, train_y = train_adj.to(device), train_y.to(device)

        val_time_X, val_gcn_X = val_time_X.to(device), val_gcn_X.to(device)
        val_adj, val_y = val_adj.to(device), val_y.to(device)

        model = CombinedCNN_GCN(
            num_classes=n_classes,
            n_chans=n_channels,
            input_size=input_size,
            GCN_hidden1=GCN_hidden1,
            GCN_hidden3=GCN_hidden3,
            dropout_rate=dropout_rate
        ).to(device)

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters for fold {fold_idx + 1}: {total_params}")

        # NOTE: Ensure your `Trainer` class unbinds 6 matrices (a1~a6) instead of 9 during training/predicting.
        trainer = Trainer(
            model=model,
            train_set=(train_time_X, train_gcn_X, train_adj, train_y),
            val_set=(val_time_X, val_gcn_X, val_adj, val_y),
            n_classes=n_classes
        )
        trainer.compile(learning_rate=learning_rate, weight_decay=1e-4)

        fold_model_path = os.path.join(model_save_dir, f'best_model_fold_{fold_idx + 1}.pt')
        history = trainer.train(
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            early_stopping_metric=early_stopping_metric,
            directory=fold_model_path,
            min_epochs=50
        )
        fold_history.append(history)

        val_preds = trainer.predict(val_time_X, val_gcn_X, val_adj)
        val_true = val_y.cpu().numpy()

        val_acc = accuracy_score(val_true, val_preds)
        val_precision = precision_score(val_true, val_preds, zero_division=0)
        val_recall = recall_score(val_true, val_preds, zero_division=0)
        val_f1 = f1_score(val_true, val_preds, zero_division=0)
        val_cm = confusion_matrix(val_true, val_preds, labels=[0, 1])

        print(f"Fold {fold_idx + 1} Validation Metrics:")
        print(f"  Accuracy: {val_acc:.4f} | Precision: {val_precision:.4f}")
        print(f"  Recall:   {val_recall:.4f} | F1 Score:  {val_f1:.4f}")

        fold_results.append({
            'fold': fold_idx + 1, 'acc': val_acc, 'precision': val_precision,
            'recall': val_recall, 'f1': val_f1, 'cm': val_cm
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_path = fold_model_path
            print(f"  ✅ Best model updated to Fold {fold_idx + 1}")

    # -------------------------- 5. Testing & Inference --------------------------
    print(f"\n=== Test Set Evaluation & Feature Extraction ===")
    test_time_X, test_gcn_X, test_y, test_adj, _ = preprocess_data(
        test_time_X, test_gcn_X, test_y, test_adj, scaler=scaler
    )
    test_time_X, test_gcn_X, test_adj = test_time_X.to(device), test_gcn_X.to(device), test_adj.to(device)

    test_preds_all, test_results, all_fusion_weights = [], [], []

    for fold_idx in range(5):
        fold_model_path = os.path.join(model_save_dir, f'best_model_fold_{fold_idx + 1}.pt')
        print(f"\nLoading Fold {fold_idx + 1} Model: {fold_model_path}")

        model = CombinedCNN_GCN(
            num_classes=n_classes, n_chans=n_channels, input_size=input_size,
            GCN_hidden1=GCN_hidden1, GCN_hidden3=GCN_hidden3, dropout_rate=dropout_rate
        ).to(device)
        model.load_state_dict(torch.load(fold_model_path, map_location=device, weights_only=True))
        model.eval()

        if fold_idx == 0:
            measure_single_sample_latency(model, test_time_X, test_gcn_X, test_adj, device, num_runs=100)

        with torch.no_grad():
            a1, a2, a3, a4, a5, a6 = torch.unbind(test_adj, dim=1)

            test_logits, fusion_weights = model(
                test_time_X, test_gcn_X, a1, a2, a3, a4, a5, a6, return_weights=True
            )
            test_preds = torch.argmax(test_logits, dim=1).cpu().numpy()
            fusion_weights_np = fusion_weights.cpu().numpy()

            # Feature Extraction for t-SNE
            cnn_features_flat = model.extract_cnn_features(test_time_X).cpu().numpy().reshape(len(test_y), -1)
            visualize_tsne(cnn_features_flat, test_true_np,
                           os.path.join(weights_dir, f'fold_{fold_idx + 1}_cnn_tsne.png'))

            gcn_features_flat = model.extract_gcn_features(test_gcn_X, a1, a2, a3, a4, a5, a6).cpu().numpy().reshape(
                len(test_y), -1)
            visualize_tsne(gcn_features_flat, test_true_np,
                           os.path.join(weights_dir, f'fold_{fold_idx + 1}_gcn_tsne.png'))

            fused_features_flat = model.extract_fused_features(test_time_X, test_gcn_X, a1, a2, a3, a4, a5,
                                                               a6).cpu().numpy().reshape(len(test_y), -1)
            visualize_tsne(fused_features_flat, test_true_np,
                           os.path.join(weights_dir, f'fold_{fold_idx + 1}_fused_tsne.png'))

        test_preds_all.append(test_preds)
        all_fusion_weights.append(fusion_weights_np)

        test_acc = accuracy_score(test_true_np, test_preds)
        test_precision = precision_score(test_true_np, test_preds, zero_division=0)
        test_recall = recall_score(test_true_np, test_preds, zero_division=0)
        test_f1 = f1_score(test_true_np, test_preds, zero_division=0)
        test_cm = confusion_matrix(test_true_np, test_preds, labels=[0, 1])

        print(f"Fold {fold_idx + 1} Test Results:")
        print(
            f"  Accuracy: {test_acc:.4f} | Precision: {test_precision:.4f} | Recall: {test_recall:.4f} | F1: {test_f1:.4f}")

        visualize_confusion_matrix(test_cm, save_path=os.path.join(model_save_dir, f'test_fold_{fold_idx + 1}_cm.png'))
        visualize_fusion_weights(fusion_weights_np,
                                 save_path=os.path.join(weights_dir, f'fold_{fold_idx + 1}_weights.png'),
                                 true_labels=test_true_np)

        cnn_mean = np.mean(fusion_weights_np[:, 0])
        gcn_mean = np.mean(fusion_weights_np[:, 1])

        test_results.append({
            'fold': fold_idx + 1, 'acc': test_acc, 'precision': test_precision,
            'recall': test_recall, 'f1': test_f1, 'cm': test_cm,
            'fusion_weights': fusion_weights_np, 'cnn_weight_mean': cnn_mean, 'gcn_weight_mean': gcn_mean
        })

    # -------------------------- 6. Final Evaluation Summaries --------------------------
    avg_test_acc = np.mean([r['acc'] for r in test_results])
    avg_test_f1 = np.mean([r['f1'] for r in test_results])
    avg_test_cm = np.mean([r['cm'] for r in test_results], axis=0).astype(int)

    test_preds_vote = np.round(np.mean(test_preds_all, axis=0)).astype(int)
    vote_acc = accuracy_score(test_true_np, test_preds_vote)
    vote_f1 = f1_score(test_true_np, test_preds_vote, zero_division=0)
    vote_cm = confusion_matrix(test_true_np, test_preds_vote, labels=[0, 1])

    print(f"\n=== Final Ensemble Results ===")
    print(f"Single Model Avg Accuracy: {avg_test_acc:.4f} | Avg F1: {avg_test_f1:.4f}")
    print(f"Voting Ensemble Accuracy:  {vote_acc:.4f} | Voting F1: {vote_f1:.4f}")

    visualize_confusion_matrix(avg_test_cm, save_path=os.path.join(model_save_dir, 'test_avg_cm.png'))
    visualize_confusion_matrix(vote_cm, save_path=os.path.join(model_save_dir, 'test_vote_cm.png'))

    # Save all output artifacts
    results_payload = {
        'test_avg_metrics': {'acc': avg_test_acc, 'f1': avg_test_f1},
        'test_vote_metrics': {'acc': vote_acc, 'f1': vote_f1},
        'scaler': scaler
    }
    np.savez(os.path.join(model_save_dir, 'fusion_model_results.npz'), **results_payload)
    print("=== Training & Evaluation Pipeline Complete ===")


if __name__ == "__main__":
    main()