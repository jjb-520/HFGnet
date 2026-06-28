import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
import numpy as np
import time
from typing import Optional, List, Dict, Tuple, Any, Union


class Trainer:
    def __init__(self,
                 model: nn.Module,
                 train_set: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
                 val_set: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
                 n_classes: int = 2):
        """
        Trainer initialization (Adapted for the 6-adjacency-matrix CombinedCNN_GCN model)
        """
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.compiled = False
        self.n_classes = n_classes

        self._unpack_and_validate_datasets(train_set, val_set)

        self.tracker: Dict[str, List[float] | np.ndarray | None] = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_precision': [], 'train_recall': [], 'train_f1': [],
            'val_precision': [], 'val_recall': [], 'val_f1': [],
            'confusion_matrix': None,
            'learning_rates': [],
            'avg_cnn_weight': [],
            'avg_gcn_weight': []
        }

        self.loss_func = self._init_weighted_loss()
        self.weight_regularization_coef = 0.01

    def extract_features(self, time_X: torch.Tensor, gcn_X: torch.Tensor, adj: torch.Tensor) -> Tuple[
        np.ndarray, np.ndarray]:
        """Extract intermediate features from CNN and GCN branches for t-SNE visualization."""
        self.model.eval()
        time_X = time_X.float().to(self.device)
        gcn_X = gcn_X.float().to(self.device)
        adj = adj.float().to(self.device)

        # Unbind 6 adjacency matrices
        a1, a2, a3, a4, a5, a6  = torch.unbind(adj, dim=1)

        cnn_features = self.model.extract_cnn_features(time_X)
        cnn_features_np = cnn_features.cpu().numpy().reshape(len(time_X), -1)

        gcn_features = self.model.extract_gcn_features(gcn_X, a1, a2, a3, a4, a5, a6 )
        gcn_features_np = gcn_features.cpu().numpy().reshape(len(time_X), -1)

        return cnn_features_np, gcn_features_np

    def _unpack_and_validate_datasets(self, train_set, val_set):
        """Validate input dimensions ensuring [B, 6, N, N] for adjacency matrices."""
        if train_set:
            self.time_X_train, self.gcn_X_train, self.adj_train, self.y_train = train_set

            assert self.gcn_X_train.dim() == 4 and self.gcn_X_train.shape[1] == 1, \
                f"GCN input must be [B,1,N,F], got {self.gcn_X_train.shape}"
            gcn_nodes = self.gcn_X_train.shape[2]

            assert self.adj_train.dim() == 4 and self.adj_train.shape[1] == 6, \
                f"Adjacency matrix must be [B,6,N,N] (6 matrices), got {self.adj_train.shape}"
            assert self.adj_train.shape[2] == gcn_nodes and self.adj_train.shape[3] == gcn_nodes, \
                f"Adjacency matrix node count must match GCN ({gcn_nodes}), got {self.adj_train.shape[2]}"
        else:
            self.time_X_train = self.gcn_X_train = self.adj_train = self.y_train = None

        if val_set:
            self.time_X_val, self.gcn_X_val, self.adj_val, self.y_val = val_set

            assert self.gcn_X_val.dim() == 4 and self.gcn_X_val.shape[1] == 1, \
                f"Val GCN input must be [B,1,N,F], got {self.gcn_X_val.shape}"
            val_gcn_nodes = self.gcn_X_val.shape[2]

            assert self.adj_val.dim() == 4 and self.adj_val.shape[1] == 6, \
                f"Val adjacency matrix must be [B,6,N,N], got {self.adj_val.shape}"
            assert self.adj_val.shape[2] == val_gcn_nodes and self.adj_val.shape[3] == val_gcn_nodes, \
                f"Val adjacency node count must match GCN ({val_gcn_nodes}), got {self.adj_val.shape[2]}"
        else:
            self.time_X_val = self.gcn_X_val = self.adj_val = self.y_val = None

    def _init_weighted_loss(self):
        """Initialize cross-entropy loss with class weights."""
        if self.y_train is None:
            return nn.CrossEntropyLoss()

        y_np = self.y_train.cpu().numpy()
        unique_classes = np.unique(y_np)
        if len(unique_classes) < self.n_classes:
            print(f"Warning: Training set has {len(unique_classes)}/{self.n_classes} classes, using equal weights")
            return nn.CrossEntropyLoss()

        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(self.n_classes),
            y=y_np
        )
        weights_tensor = torch.FloatTensor(weights).to(self.device)
        print(f"Class weights: {weights}")
        return nn.CrossEntropyLoss(weight=weights_tensor)

    def compile(self, learning_rate: float = 0.001, weight_decay: float = 0.0001) -> None:
        """Compile optimizer and learning rate scheduler."""
        self.optimizer = Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=1e-6
        )
        self.compiled = True
        print(f"Optimizer initialized: lr={learning_rate}, weight_decay={weight_decay}")
        print(f"Weight regularization coef: {self.weight_regularization_coef}")

    def train(self,
              epochs: int = 100,
              batch_size: int = 32,
              patience: int = 10,
              early_stopping_metric: str = 'val_f1',
              directory: str = 'best_model.pt',
              min_epochs: int = 50,
              extra_epochs: int = 10) -> Dict[str, Any]:
        """Main training loop with early stopping."""
        if not self.compiled:
            raise RuntimeError("Call compile() first to initialize optimizer")
        if self.time_X_train is None:
            raise ValueError("Train set not initialized: pass train_set to Trainer")

        train_loader = self._build_dataloader(
            self.time_X_train, self.gcn_X_train, self.adj_train, self.y_train, batch_size, shuffle=True
        )
        val_loader = self._build_dataloader(
            self.time_X_val, self.gcn_X_val, self.adj_val, self.y_val, batch_size, shuffle=False
        ) if self.time_X_val is not None else None

        best_metric = float('inf') if early_stopping_metric == 'val_loss' else 0.0
        wait = 0
        best_model_state = None
        early_stopped = False
        extra_counter = 0

        print(f"\nStarting training (Device: {self.device})")
        print(f"Early stopping: metric={early_stopping_metric}, patience={patience}, min epochs={min_epochs}")

        for epoch in range(epochs):
            current_epoch = epoch + 1
            epoch_start = time.time()

            # Training phase
            self.model.train()
            train_metrics = self._train_one_epoch(train_loader)
            self.tracker['train_loss'].append(train_metrics['loss'])
            self.tracker['train_acc'].append(train_metrics['acc'])
            self.tracker['train_precision'].append(train_metrics['precision'])
            self.tracker['train_recall'].append(train_metrics['recall'])
            self.tracker['train_f1'].append(train_metrics['f1'])
            self.tracker['avg_cnn_weight'].append(train_metrics['avg_cnn_weight'])
            self.tracker['avg_gcn_weight'].append(train_metrics['avg_gcn_weight'])

            current_lr = self.optimizer.param_groups[0]['lr']
            self.tracker['learning_rates'].append(current_lr)

            # Validation phase
            val_metrics = None
            if val_loader:
                self.model.eval()
                val_metrics = self._val_one_epoch(val_loader)
                self.tracker['val_loss'].append(val_metrics['val_loss'])
                self.tracker['val_acc'].append(val_metrics['val_acc'])
                self.tracker['val_precision'].append(val_metrics['val_precision'])
                self.tracker['val_recall'].append(val_metrics['val_recall'])
                self.tracker['val_f1'].append(val_metrics['val_f1'])
                self.scheduler.step(val_metrics['val_loss'])

            self._print_epoch_log(current_epoch, epochs, current_lr, train_metrics, val_metrics, epoch_start)

            # Early stopping logic
            if val_loader and not early_stopped:
                current_val_metric = val_metrics[early_stopping_metric]
                is_better = self._is_better_metric(
                    current_val_metric, best_metric, early_stopping_metric
                )
                if is_better:
                    best_metric = current_val_metric
                    best_model_state = self.model.state_dict().copy()
                    torch.save(best_model_state, directory)
                    print(f"✅ Best model saved to: {directory} ({early_stopping_metric}={best_metric:.4f})")
                    wait = 0
                else:
                    wait += 1
                    print(f"⚠️ No improvement, wait count: {wait}/{patience}")

                if wait >= patience:
                    early_stopped = True
                    print(f"⏸️ Early stopping triggered (epoch {current_epoch})")
                    if current_epoch < min_epochs:
                        print(
                            f"🔄 Min epochs not reached ({current_epoch}/{min_epochs}), continue for {extra_epochs} epochs")
                    else:
                        print(f"🛑 Min epochs reached, stop training")
                        break

            if early_stopped:
                extra_counter += 1
                if extra_counter >= extra_epochs:
                    print(f"🛑 Extra epochs completed, stop training")
                    break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"\n📥 Loaded best model weights ({early_stopping_metric}={best_metric:.4f})")

        if val_loader:
            val_preds = self.predict(self.time_X_val, self.gcn_X_val, self.adj_val)
            self.tracker['confusion_matrix'] = confusion_matrix(
                self.y_val.cpu().numpy(),
                val_preds,
                labels=np.arange(self.n_classes)
            )

        return self.tracker

    def _build_dataloader(self, time_X, gcn_X, adj, y, batch_size, shuffle):
        """Construct DataLoader."""
        if y is not None:
            dataset = torch.utils.data.TensorDataset(time_X, gcn_X, adj, y)
        else:
            dataset = torch.utils.data.TensorDataset(time_X, gcn_X, adj)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
            pin_memory=False
        )

    def _train_one_epoch(self, train_loader):
        """Train for a single epoch."""
        total_loss = 0.0
        total_cls_loss = 0.0
        all_preds = []
        all_targets = []
        all_fusion_weights = []

        for batch in train_loader:
            time_data, gcn_data, adj, target = batch

            time_data = time_data.float().to(self.device)
            gcn_data = gcn_data.float().to(self.device)
            adj = adj.float().to(self.device)
            target = target.long().to(self.device)

            # Unbind 6 adjacency matrices
            a1, a2, a3, a4, a5, a6  = torch.unbind(adj, dim=1)

            self.optimizer.zero_grad()

            logits, fusion_weights = self.model(
                time_data, gcn_data,
                a1, a2, a3, a4, a5, a6,
                return_weights=True
            )

            cls_loss = self.loss_func(logits, target)
            loss = cls_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            total_loss += loss.item() * time_data.size(0)
            total_cls_loss += cls_loss.item() * time_data.size(0)
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_fusion_weights.append(fusion_weights.detach().cpu().numpy())

        avg_loss = total_loss / len(train_loader.dataset)
        avg_cls_loss = total_cls_loss / len(train_loader.dataset)
        acc = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, zero_division=0)
        recall = recall_score(all_targets, all_preds, zero_division=0)
        f1 = f1_score(all_targets, all_preds, zero_division=0)

        all_fusion_weights_np = np.vstack(all_fusion_weights)
        avg_cnn_weight = np.mean(all_fusion_weights_np[:, 0])
        avg_gcn_weight = np.mean(all_fusion_weights_np[:, 1])

        return {
            'loss': avg_loss,
            'cls_loss': avg_cls_loss,
            'acc': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'avg_cnn_weight': avg_cnn_weight,
            'avg_gcn_weight': avg_gcn_weight
        }

    def _val_one_epoch(self, val_loader):
        """Validate for a single epoch."""
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                time_data, gcn_data, adj, target = batch

                time_data = time_data.float().to(self.device)
                gcn_data = gcn_data.float().to(self.device)
                adj = adj.float().to(self.device)
                target = target.long().to(self.device)

                # Unbind 6 adjacency matrices
                a1, a2, a3, a4, a5, a6  = torch.unbind(adj, dim=1)

                logits = self.model(
                    time_data, gcn_data,
                    a1, a2, a3, a4, a5, a6
                )

                loss = self.loss_func(logits, target)
                total_loss += loss.item() * time_data.size(0)

                all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        avg_loss = total_loss / len(val_loader.dataset)
        acc = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, zero_division=0)
        recall = recall_score(all_targets, all_preds, zero_division=0)
        f1 = f1_score(all_targets, all_preds, zero_division=0)

        return {
            'val_loss': avg_loss,
            'val_acc': acc,
            'val_precision': precision,
            'val_recall': recall,
            'val_f1': f1,
            'preds': all_preds,
            'targets': all_targets
        }

    def _print_epoch_log(self, epoch, total_epochs, lr, train_metrics, val_metrics, start_time):
        """Format and print epoch training logs."""
        epoch_time = time.time() - start_time
        log = f"\nEpoch {epoch:3d}/{total_epochs:3d} | LR: {lr:.8f} | Time: {epoch_time:.2f}s"
        log += f"\nTrain | Total loss: {train_metrics['loss']:.4f} (Classification: {train_metrics['cls_loss']:.4f} | "
        log += f"Acc: {train_metrics['acc']:.4f} | Prec: {train_metrics['precision']:.4f} | Rec: {train_metrics['recall']:.4f} | F1: {train_metrics['f1']:.4f}\n"
        log += f"      | Avg weights: CNN={train_metrics['avg_cnn_weight']:.4f}, GCN={train_metrics['avg_gcn_weight']:.4f}"
        if val_metrics:
            log += f"\nVal   | Loss: {val_metrics['val_loss']:.4f} | Acc: {val_metrics['val_acc']:.4f} | "
            log += f"Prec: {val_metrics['val_precision']:.4f} | Rec: {val_metrics['val_recall']:.4f} | F1: {val_metrics['val_f1']:.4f}"
        print(log)
        print("-" * 100)

    def _is_better_metric(self, current, best, metric_type):
        """Determine if current metric is an improvement."""
        if metric_type == 'val_loss':
            return current < best - 1e-6
        elif metric_type in ['val_acc', 'val_f1']:
            return current > best + 1e-6
        else:
            raise ValueError(f"Unsupported early stopping metric: {metric_type} (use val_loss/val_acc/val_f1)")

    def predict(self,
                time_X_test: torch.Tensor,
                gcn_X_test: torch.Tensor,
                adj_test: torch.Tensor,
                batch_size: int = 32,
                custom_model: Optional[nn.Module] = None,
                return_weights: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Inference function."""
        model = custom_model if custom_model is not None else self.model
        model.eval()
        model.to(self.device)

        assert time_X_test.dim() == 3, f"CNN input must be [B,C,T], got {time_X_test.shape}"
        assert gcn_X_test.dim() == 4 and gcn_X_test.shape[
            1] == 1, f"GCN input must be [B,1,N,F], got {gcn_X_test.shape}"
        assert adj_test.dim() == 4 and adj_test.shape[
            1] == 6, f"Adjacency matrix must be [B,6,N,N], got {adj_test.shape}"
        assert adj_test.shape[2] == gcn_X_test.shape[2], \
            f"Adjacency node count must match GCN ({gcn_X_test.shape[2]}), got {adj_test.shape[2]}"

        test_loader = self._build_dataloader(time_X_test, gcn_X_test, adj_test, None, batch_size, shuffle=False)
        all_preds = []
        all_fusion_weights = []

        with torch.no_grad():
            for batch in test_loader:
                time_data, gcn_data, adj = batch
                time_data = time_data.float().to(self.device)
                gcn_data = gcn_data.float().to(self.device)
                adj = adj.float().to(self.device)

                if time_data.size(0) == 0:
                    continue

                # Unbind 6 adjacency matrices
                a1, a2, a3, a4, a5, a6  = torch.unbind(adj, dim=1)

                if return_weights:
                    logits, fusion_weights = self.model(
                        time_data, gcn_data,
                        a1, a2, a3, a4, a5, a6,
                        return_weights=True
                    )
                    all_fusion_weights.append(fusion_weights.cpu().numpy())
                else:
                    logits = self.model(
                        time_data, gcn_data,
                        a1, a2, a3, a4, a5, a6
                    )

                all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())

        if return_weights:
            fusion_weights_np = np.vstack(all_fusion_weights)
            return np.array(all_preds), fusion_weights_np
        else:
            return np.array(all_preds)

    def evaluate(self,
                 time_X: torch.Tensor,
                 gcn_X: torch.Tensor,
                 adj: torch.Tensor,
                 y: torch.Tensor,
                 batch_size: int = 32,
                 return_weights: bool = False) -> Dict[str, Union[float, np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
        """Comprehensive evaluation function."""
        if return_weights:
            preds, fusion_weights = self.predict(
                time_X, gcn_X, adj, batch_size, return_weights=True
            )
        else:
            preds = self.predict(time_X, gcn_X, adj, batch_size)
            fusion_weights = None

        y_np = y.cpu().numpy()

        metrics = {
            'accuracy': accuracy_score(y_np, preds),
            'precision': precision_score(y_np, preds, zero_division=0, average='weighted'),
            'recall': recall_score(y_np, preds, zero_division=0, average='weighted'),
            'f1': f1_score(y_np, preds, zero_division=0, average='weighted'),
            'confusion_matrix': confusion_matrix(y_np, preds, labels=np.arange(self.n_classes))
        }

        if return_weights:
            metrics['fusion_weights'] = fusion_weights

        print("\n" + "=" * 50)
        print("Model Evaluation Results")
        print("=" * 50)
        for key, val in metrics.items():
            if key not in ['confusion_matrix', 'fusion_weights', 'gat_attention']:
                print(f"{key:12s}: {val:.4f}")
            elif key == 'confusion_matrix':
                print(f"\n{key}:")
                print(val)
            else:
                print(f"\n{key}: Shape {val.shape}")
        print("=" * 50)

        return metrics