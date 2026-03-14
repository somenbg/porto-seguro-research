"""Data preparation and evaluation for Porto Seguro Safe Driver Prediction.

Loads the Kaggle competition data, preprocesses features (handle missing values,
one-hot encode categoricals, drop noise features), and provides fixed evaluation
using the Normalized Gini Coefficient.

DO NOT MODIFY — this file is read-only for the agent.

Place train.csv at data/train.csv
"""

import os
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIME_BUDGET = 60
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "train.csv")
CACHE_PATH = os.path.join(os.path.dirname(__file__), "data", "prepared.pt")
VAL_FRACTION = 0.2
SEED = 42

# Feature groups (by naming convention in the dataset)
BINARY_COLS = [
    "ps_ind_06_bin", "ps_ind_07_bin", "ps_ind_08_bin", "ps_ind_09_bin",
    "ps_ind_10_bin", "ps_ind_11_bin", "ps_ind_12_bin", "ps_ind_13_bin",
    "ps_ind_16_bin", "ps_ind_17_bin", "ps_ind_18_bin",
    "ps_calc_15_bin", "ps_calc_16_bin", "ps_calc_17_bin",
    "ps_calc_18_bin", "ps_calc_19_bin", "ps_calc_20_bin",
]

CATEGORICAL_COLS = [
    "ps_ind_02_cat", "ps_ind_04_cat", "ps_ind_05_cat",
    "ps_car_01_cat", "ps_car_02_cat", "ps_car_03_cat",
    "ps_car_04_cat", "ps_car_05_cat", "ps_car_06_cat",
    "ps_car_07_cat", "ps_car_08_cat", "ps_car_09_cat",
    "ps_car_10_cat", "ps_car_11_cat",
]

# Calc features are known noise — drop them (except binary calc which stay)
CALC_DROP = [
    "ps_calc_01", "ps_calc_02", "ps_calc_03", "ps_calc_04",
    "ps_calc_05", "ps_calc_06", "ps_calc_07", "ps_calc_08",
    "ps_calc_09", "ps_calc_10", "ps_calc_11", "ps_calc_12",
    "ps_calc_13", "ps_calc_14",
]

CONTINUOUS_COLS = [
    "ps_ind_01", "ps_ind_03", "ps_ind_14", "ps_ind_15",
    "ps_reg_01", "ps_reg_02", "ps_reg_03",
    "ps_car_11", "ps_car_12", "ps_car_13", "ps_car_14", "ps_car_15",
]


# ---------------------------------------------------------------------------
# Normalized Gini Coefficient (official competition metric)
# ---------------------------------------------------------------------------

def _gini(actual, predicted):
    """Compute raw Gini coefficient."""
    n = len(actual)
    indices = np.argsort(-predicted)
    sorted_actual = actual[indices]
    cumulative = np.cumsum(sorted_actual)
    gini_sum = cumulative.sum() / sorted_actual.sum() - (n + 1) / 2.0
    return gini_sum / n


def normalized_gini(actual, predicted):
    """Normalized Gini coefficient — the official Kaggle metric.

    Ranges from 0 (random) to ~1 (perfect). Higher is better.
    """
    return _gini(actual, predicted) / _gini(actual, actual)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _preprocess(df):
    """Transform raw DataFrame into a feature matrix.

    Steps:
    1. Drop id, target, and calc noise features
    2. Add 'missing_count' feature (number of -1s per row)
    3. Replace -1 with NaN, then impute: mode for categoricals, mean for continuous
    4. One-hot encode categoricals (treating -1/NaN as its own category)
    5. StandardScale continuous features
    6. Return float32 numpy array
    """
    import pandas as pd

    feature_cols = BINARY_COLS + CATEGORICAL_COLS + CONTINUOUS_COLS

    data = df[feature_cols].copy()

    # Count missing values per row as a feature
    missing_count = (data == -1).sum(axis=1).values.astype(np.float32)

    # For categoricals: treat -1 as a separate category (don't replace with NaN)
    # For continuous: replace -1 with column mean
    for col in CONTINUOUS_COLS:
        data[col] = data[col].astype(np.float64)
        mask = data[col] == -1
        if mask.any():
            col_mean = data.loc[~mask, col].mean()
            data.loc[mask, col] = col_mean

    # One-hot encode categoricals
    cat_dummies = pd.get_dummies(
        data[CATEGORICAL_COLS].astype(str),
        columns=CATEGORICAL_COLS,
        drop_first=False,
        dtype=np.float32,
    )

    # Continuous features
    cont_data = data[CONTINUOUS_COLS].values.astype(np.float32)
    cont_mean = np.nanmean(cont_data, axis=0)
    cont_std = np.nanstd(cont_data, axis=0) + 1e-8
    cont_data = (cont_data - cont_mean) / cont_std

    # Binary features (already 0/1)
    bin_data = data[BINARY_COLS].values.astype(np.float32)

    # Combine
    missing_feat = missing_count.reshape(-1, 1)
    X = np.concatenate([
        cont_data,
        bin_data,
        cat_dummies.values,
        missing_feat,
    ], axis=1)

    return X.astype(np.float32)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(device="cpu", val_fraction=VAL_FRACTION):
    """Load Porto Seguro data. Returns tensors on the specified device.

    Caches preprocessed data to data/prepared.pt for fast subsequent loads.

    Returns:
        train_X, train_y, val_X, val_y, info_dict
    """
    device_t = torch.device(device) if isinstance(device, str) else device

    if os.path.exists(CACHE_PATH):
        cache = torch.load(CACHE_PATH, weights_only=True)
        train_X = cache["train_X"].to(device_t)
        train_y = cache["train_y"].to(device_t)
        val_X = cache["val_X"].to(device_t)
        val_y = cache["val_y"].to(device_t)
        info = cache["info"]
        return train_X, train_y, val_X, val_y, info

    import pandas as pd
    print(f"Loading {DATA_PATH} (first load — will cache for speed)...")

    df = pd.read_csv(DATA_PATH)
    total_rows = len(df)

    X = _preprocess(df)
    y = df["target"].values.astype(np.float32)

    # Deterministic stratified train/val split
    n = len(X)
    rng = np.random.RandomState(SEED)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    n_pos_val = max(1, int(len(pos_idx) * val_fraction))
    n_neg_val = max(1, int(len(neg_idx) * val_fraction))

    val_idx = np.concatenate([pos_idx[:n_pos_val], neg_idx[:n_neg_val]])
    train_idx = np.concatenate([pos_idx[n_pos_val:], neg_idx[n_neg_val:]])

    train_X = torch.tensor(X[train_idx])
    train_y = torch.tensor(y[train_idx])
    val_X = torch.tensor(X[val_idx])
    val_y = torch.tensor(y[val_idx])

    pos_rate_train = y[train_idx].mean()
    pos_rate_val = y[val_idx].mean()

    info = {
        "num_features": X.shape[1],
        "num_classes": 2,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "pos_rate_train": float(pos_rate_train),
        "pos_rate_val": float(pos_rate_val),
        "total_rows": total_rows,
        "pos_weight": float((1 - pos_rate_train) / pos_rate_train),
    }

    torch.save({
        "train_X": train_X, "train_y": train_y,
        "val_X": val_X, "val_y": val_y,
        "info": info,
    }, CACHE_PATH)
    print(f"  Cached to {CACHE_PATH}")
    print(f"  Features: {info['num_features']}, Train: {info['n_train']:,}, Val: {info['n_val']:,}")
    print(f"  Positive rate — train: {pos_rate_train:.4f}, val: {pos_rate_val:.4f}")

    train_X = train_X.to(device_t)
    train_y = train_y.to(device_t)
    val_X = val_X.to(device_t)
    val_y = val_y.to(device_t)

    return train_X, train_y, val_X, val_y, info


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, val_X, val_y, autocast_ctx=None):
    """Evaluate model on validation set using Normalized Gini Coefficient.

    Model should output raw logits (single output neuron, no sigmoid).
    Returns normalized_gini score (higher is better, range ~0 to 1).
    """
    import contextlib
    ctx = autocast_ctx if autocast_ctx is not None else contextlib.nullcontext()
    model.eval()

    with ctx:
        logits = model(val_X)

    # Convert logits to probabilities
    probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    actual = val_y.cpu().numpy()

    return normalized_gini(actual, probs)
