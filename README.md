# Porto Seguro Safe Driver Prediction — AutoResearch

Autonomous ML research on predicting whether a driver will file an auto
insurance claim next year. A notoriously hard binary classification problem
with extreme class imbalance and anonymized features.

Built on the [autoresearch](https://github.com/somenbg/autoresearch) framework.

## Dataset

[Porto Seguro's Safe Driver Prediction](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction)
— 595K rows, 57 anonymized features, 3.6% positive rate.

**Setup**: Download `train.csv` from Kaggle and place it at `data/train.csv`.

## Quick Start

```bash
uv sync
uv run train.py                    # single training run
uv run ares research --budget 10   # autonomous research loop (10 experiments)
uv run ares dashboard              # live experiment dashboard
uv run ares history                # browse experiment history
```

## Baseline

val_gini ~0.125 with a 3-layer MLP (96K params) on Apple M2 in 60 seconds.
Kaggle top solutions achieve ~0.29 — plenty of room for the research loop.
