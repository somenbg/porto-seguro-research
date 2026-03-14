"""
Porto Seguro Safe Driver Prediction — AutoResearch training script.
Usage: uv run train.py

Binary classification: predict probability of insurance claim.
Metric: Normalized Gini Coefficient (higher is better).
The agent modifies this file to improve val_gini.
"""

import time

import torch
import torch.nn as nn

from ar.platform import detect_platform
from ar.platform_utils import (
    synchronize,
    get_autocast_context,
    get_peak_memory_mb,
    seed_everything,
    should_compile,
)
from prepare import load_data, evaluate, TIME_BUDGET

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

HIDDEN_DIMS = [256, 128, 64]
DROPOUT = 0.15
LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 512
ACTIVATION = "relu"
USE_POS_WEIGHT = True       # weight positive class in BCE loss
USE_BATCH_NORM = False       # batch normalization between layers
LABEL_SMOOTHING = 0.0        # smooth labels: y = y*(1-s) + 0.5*s

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout, activation, use_bn=False):
        super().__init__()
        act_fns = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU, "tanh": nn.Tanh}
        act_class = act_fns.get(activation, nn.ReLU)

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act_class())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

platform = detect_platform()
device = torch.device(platform.device)
autocast_ctx = get_autocast_context(platform)
seed_everything(42, device)

print(f"Platform: {platform.device_name} ({platform.device})")
print(f"Memory: {platform.usable_memory_mb:,} MB usable")
print(f"Time budget: {TIME_BUDGET}s")

train_X, train_y, val_X, val_y, info = load_data(device=device)
input_dim = info["num_features"]
print(f"Task: binary classification | Features: {input_dim}")
print(f"Train: {info['n_train']:,} (pos rate: {info['pos_rate_train']:.4f})")
print(f"Val:   {info['n_val']:,} (pos rate: {info['pos_rate_val']:.4f})")

model = MLP(input_dim, HIDDEN_DIMS, DROPOUT, ACTIVATION, USE_BATCH_NORM).to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {num_params:,}")

if should_compile(platform):
    model = torch.compile(model)

# Loss with optional positive class weighting
loss_kwargs = {}
if USE_POS_WEIGHT:
    pw = torch.tensor([info["pos_weight"]], device=device)
    loss_kwargs["pos_weight"] = pw
    print(f"Pos weight: {info['pos_weight']:.1f}")

criterion = nn.BCEWithLogitsLoss(**loss_kwargs)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start = time.time()
step = 0
epoch = 0
n_train = len(train_X)

while True:
    model.train()
    epoch += 1
    perm = torch.randperm(n_train, device=device)

    for i in range(0, n_train, BATCH_SIZE):
        idx = perm[i : i + BATCH_SIZE]
        xb = train_X[idx]
        yb = train_y[idx]

        if LABEL_SMOOTHING > 0:
            yb = yb * (1.0 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING

        with autocast_ctx:
            logits = model(xb).squeeze(-1)
            loss = criterion(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1

    synchronize(device)
    elapsed = time.time() - t_start
    if elapsed >= TIME_BUDGET:
        break

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

model.eval()
val_metric = evaluate(model, val_X, val_y, autocast_ctx)
peak_mem = get_peak_memory_mb(device)
t_end = time.time()

print("---")
print(f"val_gini:         {val_metric:.6f}")
print(f"training_seconds: {elapsed:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_memory_mb:   {peak_mem:.1f}")
print(f"num_steps:        {step}")
print(f"num_epochs:       {epoch}")
print(f"num_params:       {num_params}")
print(f"device:           {platform.device}")
print(f"device_name:      {platform.device_name}")
