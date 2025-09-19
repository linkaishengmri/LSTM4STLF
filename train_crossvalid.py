# Notebook: train_lstm_with_month_pytorch_cv_final.py
# Requirements: pytorch, numpy, scipy, scikit-learn, matplotlib, seaborn
# In terminal: pip install torch scipy numpy matplotlib seaborn scikit-learn

import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold

# ---------------------------
# Config
# ---------------------------
MAT_PATH = "load_dataset.mat"
BATCH_SIZE = 32
EPOCHS = 23
LR = 1e-3
HIDDEN_SIZE = 128
NUM_LAYERS = 1
DROPOUT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
PRINT_EVERY = 1
EPS = 1e-6  # avoid div-by-zero in MAPE
K_FOLDS = 5

BEST_MODEL_PREFIX = "best_lstm_month_fold"  # will append fold index
FINAL_MODEL_PATH = "final_lstm_month_alltrain.pth"

# make dir for plots
os.makedirs("analysis_plots", exist_ok=True)

# ---------------------------
# Plot style
# ---------------------------
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")

# ---------------------------
# Seed (reproducible)
# ---------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(SEED)

# ---------------------------
# Load .mat (given var names)
# ---------------------------
import h5py

with h5py.File(MAT_PATH, "r") as f:
    def load_array(name):
        return np.array(f[name]).T  # transpose to match MATLAB dims

    X_train = load_array("X_train")            # (168, 52, 307)
    X_train_month = load_array("X_train_month")# (4, 52, 307)
    Y_train = load_array("Y_train")            # (307, 168)

    X_test = load_array("X_test")              # (168, 52, 104)
    X_test_month = load_array("X_test_month")  # (4, 52, 104)
    Y_test = load_array("Y_test")              # (104, 168)

print("X_train", X_train.shape)
print("X_train_month", X_train_month.shape)
print("Y_train", Y_train.shape)
print("X_test", X_test.shape)
print("X_test_month", X_test_month.shape)
print("Y_test", Y_test.shape)

# ---------------------------
# Reorder to (n_samples, seq_len=52, input_size=168)
# ---------------------------
def reorder_X(X):
    return np.transpose(X, (2, 1, 0))

X_train_proc = reorder_X(X_train).astype(np.float32)  # (307, 52, 168)
X_test_proc = reorder_X(X_test).astype(np.float32)    # (104, 52, 168)

# Process month: take target week's month vector (last week)
def extract_month(X_month):
    return np.transpose(X_month[:, -1, :], (1, 0)).astype(np.float32)

X_train_month_proc = extract_month(X_train_month)  # (307, 4)
X_test_month_proc = extract_month(X_test_month)    # (104, 4)

Y_train_proc = Y_train.astype(np.float32)   # (307, 168)
Y_test_proc = Y_test.astype(np.float32)     # (104, 168)

print("After preprocess shapes:")
print("X_train_proc", X_train_proc.shape)
print("X_train_month_proc", X_train_month_proc.shape)
print("Y_train_proc", Y_train_proc.shape)

# ---------------------------
# Simple normalization (use train stats)
# ---------------------------
train_mean = X_train_proc.mean()
train_std = X_train_proc.std()
print(f"Global train mean/std = {train_mean:.4f} / {train_std:.4f}")
X_train_proc = (X_train_proc - train_mean) / (train_std + 1e-9)
X_test_proc = (X_test_proc - train_mean) / (train_std + 1e-9)

# ---------------------------
# PyTorch Dataset
# ---------------------------
class LoadDataset(torch.utils.data.Dataset):
    def __init__(self, X_seq, X_month, Y):
        assert X_seq.shape[0] == X_month.shape[0] == Y.shape[0]
        self.X_seq = torch.from_numpy(X_seq)
        self.X_month = torch.from_numpy(X_month)
        self.Y = torch.from_numpy(Y)

    def __len__(self):
        return self.X_seq.shape[0]

    def __getitem__(self, idx):
        return self.X_seq[idx], self.X_month[idx], self.Y[idx]

full_train_ds = LoadDataset(X_train_proc, X_train_month_proc, Y_train_proc)
test_ds = LoadDataset(X_test_proc, X_test_month_proc, Y_test_proc)

test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# ---------------------------
# Model
# ---------------------------
class LSTMWithMonth(nn.Module):
    def __init__(self, input_size=168, hidden_size=256, num_layers=2, month_dim=4, dropout=0.2, out_dim=168):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=False)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size + month_dim, 512),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, out_dim)
        )

    def forward(self, x_seq, x_month):
        outputs, (h_n, c_n) = self.lstm(x_seq)
        last_hidden = h_n[-1]
        cat = torch.cat([last_hidden, x_month], dim=1)
        out = self.fc(cat)
        return out

# ---------------------------
# Metrics
# ---------------------------

def mape_torch(y_true, y_pred, eps=1e-6):
    denom = torch.abs(y_true)
    denom = torch.where(denom < eps, torch.full_like(denom, eps), denom)
    return torch.mean(torch.abs((y_true - y_pred) / denom)) * 100.0


def safe_mape_np(y_true, y_pred, eps=1e-6):
    denom = np.abs(y_true)
    denom[denom < eps] = eps
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0

# ---------------------------
# K-Fold CV (train set)
# ---------------------------
kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

fold_test_metrics = []  # (mape, rmse, mae, r2) per fold on test set
fold_best_val_losses = []
fold_best_val_mapes = []

all_folds_train_losses = []
all_folds_train_mapes = []
all_folds_val_losses = []
all_folds_val_mapes = []

for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(full_train_ds)))):
    print("" + "="*60)
    print(f"Starting fold {fold+1}/{K_FOLDS}  -- Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
    print("="*60)

    train_subset = Subset(full_train_ds, train_idx)
    val_subset = Subset(full_train_ds, val_idx)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

    set_seed(SEED + fold)
    model = LSTMWithMonth(input_size=168, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, month_dim=4, dropout=DROPOUT, out_dim=168)
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_val_mape = None
    best_model_path = f"{BEST_MODEL_PREFIX}{fold}.pth"

    fold_train_losses = []
    fold_train_mapes = []
    fold_val_losses = []
    fold_val_mapes = []

    for epoch in range(1, EPOCHS+1):
        model.train()
        epoch_losses = []
        epoch_mapes = []
        for X_seq, X_month, Y in train_loader:
            X_seq = X_seq.to(DEVICE)
            X_month = X_month.to(DEVICE)
            Y = Y.to(DEVICE)

            optimizer.zero_grad()
            preds = model(X_seq, X_month)
            loss = criterion(preds, Y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_losses.append(loss.item())
            epoch_mapes.append(mape_torch(Y, preds).item())

        # validation
        model.eval()
        val_losses_epoch = []
        val_mapes_epoch = []
        with torch.no_grad():
            for X_seq, X_month, Y in val_loader:
                X_seq = X_seq.to(DEVICE)
                X_month = X_month.to(DEVICE)
                Y = Y.to(DEVICE)
                preds = model(X_seq, X_month)
                val_losses_epoch.append(criterion(preds, Y).item())
                val_mapes_epoch.append(mape_torch(Y, preds).item())

        # monitor on unchanged test set
        test_losses_epoch = []
        test_mapes_epoch = []
        with torch.no_grad():
            for X_seq, X_month, Y in test_loader:
                X_seq = X_seq.to(DEVICE)
                X_month = X_month.to(DEVICE)
                Y = Y.to(DEVICE)
                preds = model(X_seq, X_month)
                test_losses_epoch.append(criterion(preds, Y).item())
                test_mapes_epoch.append(mape_torch(Y, preds).item())

        train_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        train_mape = np.mean(epoch_mapes) if epoch_mapes else 0.0
        val_loss = np.mean(val_losses_epoch) if val_losses_epoch else 0.0
        val_mape = np.mean(val_mapes_epoch) if val_mapes_epoch else 0.0
        test_loss = np.mean(test_losses_epoch) if test_losses_epoch else 0.0
        test_mape = np.mean(test_mapes_epoch) if test_mapes_epoch else 0.0

        fold_train_losses.append(train_loss)
        fold_train_mapes.append(train_mape)
        fold_val_losses.append(val_loss)
        fold_val_mapes.append(val_mape)

        scheduler.step(val_loss)

        # save best model by val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_mape = val_mape
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss': val_loss,
            }, best_model_path)

        if epoch % PRINT_EVERY == 0:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Fold {fold+1} Epoch {epoch}/{EPOCHS} "
                  f"TrainLoss={train_loss:.6f} TrainMAPE={train_mape:.3f}% | "
                  f"ValLoss={val_loss:.6f} ValMAPE={val_mape:.3f}% | "
                  f"TestLoss={test_loss:.6f} TestMAPE={test_mape:.3f}%")

    # record fold-best validation metrics
    fold_best_val_losses.append(best_val_loss)
    fold_best_val_mapes.append(best_val_mape)

    all_folds_train_losses.append(fold_train_losses)
    all_folds_train_mapes.append(fold_train_mapes)
    all_folds_val_losses.append(fold_val_losses)
    all_folds_val_mapes.append(fold_val_mapes)

    # evaluate this fold's best model on test set
    checkpoint = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    all_preds = []
    all_trues = []
    with torch.no_grad():
        for X_seq, X_month, Y in test_loader:
            X_seq = X_seq.to(DEVICE)
            X_month = X_month.to(DEVICE)
            preds = model(X_seq, X_month).cpu().numpy()
            all_preds.append(preds)
            all_trues.append(Y.numpy())
    all_preds = np.vstack(all_preds)
    all_trues = np.vstack(all_trues)

    final_mape = safe_mape_np(all_trues, all_preds)
    final_rmse = np.sqrt(mean_squared_error(all_trues.flatten(), all_preds.flatten()))
    final_mae = mean_absolute_error(all_trues.flatten(), all_preds.flatten())
    final_r2 = r2_score(all_trues.flatten(), all_preds.flatten())

    fold_test_metrics.append((final_mape, final_rmse, final_mae, final_r2))

    print(f"Fold {fold+1} Test Performance -- MAPE: {final_mape:.4f}%, RMSE: {final_rmse:.4f}, MAE: {final_mae:.4f}, R2: {final_r2:.4f}")

# ---------------------------
# Overall val perf = mean of per-fold best val metrics
# ---------------------------
fold_best_val_losses = np.array(fold_best_val_losses)
fold_best_val_mapes = np.array(fold_best_val_mapes)
print("" + "="*60)
print("Overall validation performance (average of per-fold best val metrics):")
print(f"Val Loss (MSE) mean: {fold_best_val_losses.mean():.6f}  std: {fold_best_val_losses.std():.6f}")
print(f"Val MAPE (%)   mean: {fold_best_val_mapes.mean():.4f}  std: {fold_best_val_mapes.std():.4f}")
print("="*60)

# ---------------------------
# Print per-fold test results (mean ± std)
# ---------------------------
fold_test_metrics = np.array(fold_test_metrics)
mean_metrics = fold_test_metrics.mean(axis=0)
std_metrics = fold_test_metrics.std(axis=0)

print("" + "="*60)
print("Cross-Validation Test-set Results (per-fold best models evaluated on the same test set):")
print(f"MAPE: {mean_metrics[0]:.4f}% ± {std_metrics[0]:.4f}")
print(f"RMSE: {mean_metrics[1]:.4f} ± {std_metrics[1]:.4f}")
print(f"MAE : {mean_metrics[2]:.4f} ± {std_metrics[2]:.4f}")
print(f"R²  : {mean_metrics[3]:.4f} ± {std_metrics[3]:.4f}")
print("="*60)

# ---------------------------
# Train final model on full train+val (for reporting/deploy)
# ---------------------------
print("Training final model on the entire training+validation set (same hyperparameters)...")
set_seed(SEED)
final_model = LSTMWithMonth(input_size=168, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, month_dim=4, dropout=DROPOUT, out_dim=168)
final_model = final_model.to(DEVICE)
optimizer = torch.optim.Adam(final_model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)
criterion = nn.MSELoss()

full_train_loader = DataLoader(full_train_ds, batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(1, EPOCHS+1):
    final_model.train()
    epoch_losses = []
    epoch_mapes = []
    for X_seq, X_month, Y in full_train_loader:
        X_seq = X_seq.to(DEVICE)
        X_month = X_month.to(DEVICE)
        Y = Y.to(DEVICE)

        optimizer.zero_grad()
        preds = final_model(X_seq, X_month)
        loss = criterion(preds, Y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(final_model.parameters(), 5.0)
        optimizer.step()

        epoch_losses.append(loss.item())
        epoch_mapes.append(mape_torch(Y, preds).item())

    # train metrics for this epoch
    train_loss = np.mean(epoch_losses) if epoch_losses else 0.0
    train_mape = np.mean(epoch_mapes) if epoch_mapes else 0.0

    # test eval for monitoring
    final_model.eval()
    test_losses_epoch = []
    test_mapes_epoch = []
    with torch.no_grad():
        for X_seq, X_month, Y in test_loader:
            X_seq = X_seq.to(DEVICE)
            X_month = X_month.to(DEVICE)
            Y = Y.to(DEVICE)
            preds = final_model(X_seq, X_month)
            test_losses_epoch.append(criterion(preds, Y).item())
            test_mapes_epoch.append(mape_torch(Y, preds).item())
    test_loss = np.mean(test_losses_epoch) if test_losses_epoch else 0.0
    test_mape = np.mean(test_mapes_epoch) if test_mapes_epoch else 0.0

    scheduler.step(test_loss)

    if epoch % PRINT_EVERY == 0:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FinalModel Epoch {epoch}/{EPOCHS} "
              f"TrainLoss={train_loss:.6f} TrainMAPE={train_mape:.3f}% | "
              f"TestLoss={test_loss:.6f} TestMAPE={test_mape:.3f}%")

# Save final model
torch.save({
    'epoch': EPOCHS,
    'model_state': final_model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
}, FINAL_MODEL_PATH)

# Compute final model metrics on training set (training performance)
final_model.eval()
train_preds = []
train_trues = []
with torch.no_grad():
    for X_seq, X_month, Y in DataLoader(full_train_ds, batch_size=BATCH_SIZE, shuffle=False):
        X_seq = X_seq.to(DEVICE)
        X_month = X_month.to(DEVICE)
        preds = final_model(X_seq, X_month).cpu().numpy()
        train_preds.append(preds)
        train_trues.append(Y.numpy())
train_preds = np.vstack(train_preds)
train_trues = np.vstack(train_trues)

train_mape_final = safe_mape_np(train_trues, train_preds)
train_rmse_final = np.sqrt(mean_squared_error(train_trues.flatten(), train_preds.flatten()))
train_mae_final = mean_absolute_error(train_trues.flatten(), train_preds.flatten())
train_r2_final = r2_score(train_trues.flatten(), train_preds.flatten())


print("Final model performance on the ENTIRE training+validation set (training performance):")
print(f"MAPE: {train_mape_final:.4f}% | RMSE: {train_rmse_final:.4f} | MAE: {train_mae_final:.4f} | R2: {train_r2_final:.4f}")
print("="*60)

# Evaluate final model on test set
all_preds = []
all_trues = []
with torch.no_grad():
    for X_seq, X_month, Y in test_loader:
        X_seq = X_seq.to(DEVICE)
        X_month = X_month.to(DEVICE)
        preds = final_model(X_seq, X_month).cpu().numpy()
        all_preds.append(preds)
        all_trues.append(Y.numpy())
all_preds = np.vstack(all_preds)
all_trues = np.vstack(all_trues)

final_mape = safe_mape_np(all_trues, all_preds)
final_rmse = np.sqrt(mean_squared_error(all_trues.flatten(), all_preds.flatten()))
final_mae = mean_absolute_error(all_trues.flatten(), all_preds.flatten())
final_r2 = r2_score(all_trues.flatten(), all_preds.flatten())


print("Final model performance on the TEST set:")
print(f"MAPE: {final_mape:.4f}%")
print(f"RMSE: {final_rmse:.4f}")
print(f"MAE: {final_mae:.4f}")
print(f"R²: {final_r2:.4f}")

# ---------------------------
# Added: summary of three MAPEs (train, cv, test)
# ---------------------------
# fold_best_val_mapes collected during CV (best val MAPE per fold)
# mean_metrics, std_metrics hold per-fold best models' test results
cv_val_mape_mean = float(fold_best_val_mapes.mean()) if len(fold_best_val_mapes) > 0 else float('nan')
cv_val_mape_std = float(fold_best_val_mapes.std()) if len(fold_best_val_mapes) > 0 else float('nan')

cv_test_mape_mean = float(mean_metrics[0]) if 'mean_metrics' in globals() and len(mean_metrics) > 0 else float('nan')
cv_test_mape_std = float(std_metrics[0]) if 'std_metrics' in globals() and len(std_metrics) > 0 else float('nan')

print("="*60)
print("SUMMARY OF MODEL MAPEs:")
print(f"Train MAPE (final model on full training set): {train_mape_final:.4f}%")
print(f"CV Val MAPE (average of per-fold best validation MAPE): {cv_val_mape_mean:.4f}% ± {cv_val_mape_std:.4f}%")
print(f"Test MAPE (final model on test set): {final_mape:.4f}%")
print(f"Test MAPE (CV average of fold-best models on test set): {cv_test_mape_mean:.4f}% ± {cv_test_mape_std:.4f}%")
print("="*60)

# ---------------------------
# Plots & save analysis results (added)
# ---------------------------
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("analysis_plots", exist_ok=True)

# 1) CV train/val curves (mean ± std) if records exist
try:
    train_losses_arr = np.array(all_folds_train_losses)  # shape (n_folds, epochs)
    val_losses_arr = np.array(all_folds_val_losses)
    train_mapes_arr = np.array(all_folds_train_mapes)
    val_mapes_arr = np.array(all_folds_val_mapes)

    epochs = train_losses_arr.shape[1]
    x = np.arange(1, epochs+1)

    # Loss mean+std
    mean_train_loss = train_losses_arr.mean(axis=0)
    std_train_loss = train_losses_arr.std(axis=0)
    mean_val_loss = val_losses_arr.mean(axis=0)
    std_val_loss = val_losses_arr.std(axis=0)

    plt.figure(figsize=(10,6))
    plt.plot(x, mean_train_loss, label='CV Mean Train Loss', linewidth=2)
    plt.fill_between(x, mean_train_loss - std_train_loss, mean_train_loss + std_train_loss, alpha=0.2)
    plt.plot(x, mean_val_loss, label='CV Mean Val Loss', linewidth=2)
    plt.fill_between(x, mean_val_loss - std_val_loss, mean_val_loss + std_val_loss, alpha=0.2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Cross-Validation Loss (mean ± std)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig('analysis_plots/cv_loss_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    # MAPE mean+std
    mean_train_mape = train_mapes_arr.mean(axis=0)
    std_train_mape = train_mapes_arr.std(axis=0)
    mean_val_mape = val_mapes_arr.mean(axis=0)
    std_val_mape = val_mapes_arr.std(axis=0)

    plt.figure(figsize=(5,3))
    plt.plot(x, mean_train_mape, label='CV Mean Train MAPE', linewidth=2)
    plt.fill_between(x, mean_train_mape - std_train_mape, mean_train_mape + std_train_mape, alpha=0.2)
    plt.plot(x, mean_val_mape, label='CV Mean Val MAPE', linewidth=2)
    plt.fill_between(x, mean_val_mape - std_val_mape, mean_val_mape + std_val_mape, alpha=0.2)
    plt.xlabel('Epoch')
    plt.ylabel('MAPE (%)')
    plt.title('Cross-Validation MAPE (mean ± std)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig('analysis_plots/cv_mape_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
except Exception as e:
    print("Skipping CV-curve plots (missing records or shape mismatch). Error:", e)


# 2) Final model test analysis (uses all_trues, all_preds)
# Ensure all_trues/all_preds exist
try:
    # per-hour MAPE
    per_hour_mape = np.mean(np.abs((all_trues - all_preds) / (np.abs(all_trues) + 1e-6)), axis=0) * 100.0

    plt.figure(figsize=(6,3))
    plt.bar(np.arange(1, 169), per_hour_mape, alpha=0.8)
    plt.xlabel('Hour')
    plt.ylabel('MAPE (%)')
    plt.title('MAPE by Hour (Final Model on Test Set)')
    plt.xticks(range(1, 169, 24))
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.savefig('analysis_plots/mape_by_hour_final.png', dpi=300, bbox_inches='tight')
    plt.close()

    # True vs Pred scatter (sample up to 1000 points)
    flat_true = all_trues.flatten()
    flat_pred = all_preds.flatten()
    N = len(flat_true)
    rng = np.random.default_rng(SEED)
    if N > 1000:
        indices = rng.choice(N, size=1000, replace=False)
        sample_true = flat_true[indices]
        sample_pred = flat_pred[indices]
    else:
        sample_true = flat_true
        sample_pred = flat_pred

    plt.figure(figsize=(4,4))
    plt.scatter(sample_true, sample_pred, alpha=0.6, s=20)
    min_val = min(np.min(sample_true), np.min(sample_pred))
    max_val = max(np.max(sample_true), np.max(sample_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(f'True vs Predicted Values (R² = {final_r2:.4f})')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig('analysis_plots/true_vs_pred_final.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Error distribution
    errors = flat_true - flat_pred
    plt.figure(figsize=(5,3))
    plt.hist(errors, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Error (True - Predicted)')
    plt.ylabel('Frequency')
    plt.title('Error Distribution (Final Model on Test Set)')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.savefig('analysis_plots/error_distribution_final.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Relative error percent distribution
    relative_errors = errors / (np.abs(flat_true) + 1e-6) * 100
    plt.figure(figsize=(5,3))
    plt.hist(relative_errors, bins=50, range=(-100,100), alpha=0.7, edgecolor='black')
    plt.xlabel('Relative Error (%)')
    plt.ylabel('Frequency')
    plt.title('Relative Error Distribution (%) (Final Model on Test Set)')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.savefig('analysis_plots/relative_error_distribution_final.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Sample prediction comparisons (up to 5)
    n_samples = min(5, all_trues.shape[0])
    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 3*n_samples))
    if n_samples == 1:
        axes = [axes]
    start = 15  # keep start index as in original
    for i in range(n_samples):
        idx = i + start if (i + start) < all_trues.shape[0] else i
        axes[i].plot(all_trues[idx], label='True', linewidth=2)
        axes[i].plot(all_preds[idx], label='Predicted', linewidth=2, linestyle='--')
        axes[i].set_ylabel('Value')
        axes[i].set_title(f'Sample {i+1} (Idx {idx}) - MAPE: {safe_mape_np(all_trues[idx], all_preds[idx]):.2f}%')
        axes[i].legend()
        axes[i].grid(True, linestyle='--', alpha=0.3)
    plt.xlabel('Hour')
    plt.tight_layout()
    plt.savefig('analysis_plots/sample_predictions_final.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Metrics summary bar
    metrics = ['MAPE (%)', 'RMSE', 'MAE', 'R²']
    values = [final_mape, final_rmse, final_mae, final_r2]
    plt.figure(figsize=(8,6))
    bars = plt.bar(metrics, values)
    plt.ylabel('Value')
    plt.title('Final Model Performance (Test Set)')
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{value:.4f}', ha='center', va='bottom')
    plt.savefig('analysis_plots/metrics_summary_final.png', dpi=300, bbox_inches='tight')
    plt.close()

    # per-sample MAPE, save highest/lowest MAPE samples
    sample_mapes = np.array([safe_mape_np(all_trues[i], all_preds[i]) for i in range(len(all_trues))])
    max_mape_idx = int(np.argmax(sample_mapes))
    min_mape_idx = int(np.argmin(sample_mapes))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5))
    ax1.plot(all_trues[max_mape_idx], label='True', linewidth=2)
    ax1.plot(all_preds[max_mape_idx], label='Pred', linewidth=2, linestyle='--')
    ax1.set_ylabel('Value')
    ax1.set_title(f'Highest MAPE Sample (Idx {max_mape_idx}) - MAPE: {sample_mapes[max_mape_idx]:.2f}%')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.3)

    ax2.plot(all_trues[min_mape_idx], label='True', linewidth=2)
    ax2.plot(all_preds[min_mape_idx], label='Pred', linewidth=2, linestyle='--')
    ax2.set_xlabel('Hour')
    ax2.set_ylabel('Value')
    ax2.set_title(f'Lowest MAPE Sample (Idx {min_mape_idx}) - MAPE: {sample_mapes[min_mape_idx]:.2f}%')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig('analysis_plots/extreme_mape_samples_final.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("All analysis plots saved to analysis_plots (final model)")
    print(f"Highest MAPE sample idx: {max_mape_idx}, MAPE: {sample_mapes[max_mape_idx]:.2f}%")
    print(f"Lowest MAPE sample idx: {min_mape_idx}, MAPE: {sample_mapes[min_mape_idx]:.2f}%")

except Exception as e:
    print("Skipping final-model plots due to error:", e)
