# Porto Seguro Safe Driver Prediction — Research Program

## Task
Binary classification: predict whether a driver will file an auto insurance
claim next year. This is a notoriously hard problem — top Kaggle solutions
achieved Normalized Gini ~0.29.

## Data
- 595K rows, 57 anonymized features (binary, categorical, continuous)
- Extreme class imbalance: only 3.6% positive (filed a claim)
- Features grouped by prefix: `ind` (individual), `reg` (registration),
  `car` (vehicle), `calc` (calculated — known noise, dropped in preprocessing)
- Missing values coded as -1, handled in prepare.py
- One-hot encoded categoricals → ~220 total input features after preprocessing

## Rules
1. Only modify `train.py`.
2. Do NOT modify `prepare.py` (data loading, preprocessing, and eval are fixed).
3. Metric: `val_gini` (Normalized Gini Coefficient, maximize).
   Must be printed as `val_gini: <value>`.
4. Time budget: 60 seconds per experiment.
5. No new pip dependencies.
6. All changes must be valid Python.

## Key Challenges
- Extreme class imbalance makes naive training difficult
- Anonymized features prevent domain-guided engineering
- Weak signal: even strong models struggle past Gini ~0.29
- Overfitting risk with 220+ features and subtle patterns

## Research Directions
- **Architecture**: deeper/wider networks, residual connections, skip connections
- **Batch normalization**: critical for deep tabular networks
- **Activation functions**: GELU, SiLU, Mish may help with subtle patterns
- **Class imbalance**: pos_weight tuning, focal loss, oversampling in batches
- **Regularization**: dropout schedules, weight decay sweeps, label smoothing
- **Optimizers**: AdamW, SGD with warm restarts, cosine annealing
- **Learning rate**: warmup + cosine decay, one-cycle policy
- **Ensemble tricks**: snapshot ensembling within the time budget
- **Feature interaction**: learned embeddings for categorical groups, attention
