"""Functions that run inside the worker processes (CS4207 Week 9 lab)."""
import os
os.environ["OMP_NUM_THREADS"] = "1"        # one maths thread per process, so workers do not fight over cores
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def prepare_shard(path, shard_id, out_dir, features, sample_frac=1.0, threshold=20.0, seed=4207):
    """Clean one monthly file, split it 70/15/15 and save it as .npy files."""
    cols = ["fare_amount", "trip_distance", "passenger_count", "tpep_pickup_datetime", "tpep_dropoff_datetime"]
    t = pd.read_parquet(path, columns=cols)
    # Column pruning (meter errors e.g. zero-distance trips, zero/negative duration trips, etc.)
    t = t[(t.fare_amount > 0) & (t.trip_distance > 0) & (t.trip_distance < 100) & (t.tpep_dropoff_datetime > t.tpep_pickup_datetime)].copy()
    # Sort values by time
    t = t.sort_values("tpep_pickup_datetime").reset_index(drop=True)
    t["pickup_hour"] = t.tpep_pickup_datetime.dt.hour
    # Drop invalid values on the caller's features list
    t = t.dropna(subset=features)
    X = t[features].to_numpy(dtype=float)
    # Assign truth labels to dataset
    y = (t.fare_amount > threshold).to_numpy(dtype=float)

    # Time-based train/test split
    n_total = len(y)
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)
    parts = {
        "train": np.arange(0, n_train),
        "val": np.arange(n_train, n_train + n_val),
        "test": np.arange(n_train + n_val, n_total)
    }
    for split, idx in parts.items():
        np.save(f"{out_dir}/s{shard_id:02d}_{split}_rawX.npy", X[idx])
        np.save(f"{out_dir}/s{shard_id:02d}_{split}_y.npy", y[idx])

    Xtr = X[parts["train"]]
    return {"n": len(Xtr), "sum": Xtr.sum(axis=0), "sumsq": (Xtr ** 2).sum(axis=0),
            "rows_kept": n_total}

def scale_shard(shard_id, out_dir, mean, std):
    """Standardise one shard using the mean and std from the TRAINING data only."""
    for split in ("train", "val", "test"):
        raw = np.load(f"{out_dir}/s{shard_id:02d}_{split}_rawX.npy")
        np.save(f"{out_dir}/s{shard_id:02d}_{split}_X.npy", (raw - mean) / std)
        os.remove(f"{out_dir}/s{shard_id:02d}_{split}_rawX.npy")   # saves disk space on the shared server


def shard_gradient(shard_id, out_dir, w, b):
    """Data parallelism: the same gradient sum on one shard's training rows."""
    X = np.load(f"{out_dir}/s{shard_id:02d}_train_X.npy", mmap_mode="r")
    y = np.load(f"{out_dir}/s{shard_id:02d}_train_y.npy", mmap_mode="r")
    p = sigmoid(X @ w + b)
    err = p - y                                  # p_i - y_i
    p = np.clip(p, 1e-9, 1 - 1e-9)
    loss_sum = -(y * np.log(p) + (1 - y) * np.log(1 - p)).sum()
    return X.T @ err, err.sum(), loss_sum, len(y)   # G_k for w, for b, loss, n_k


def train_one_setting(alpha, lam, out_dir, iters=500, tol=1e-6):
    """Task parallelism: one complete training run for one (alpha, lambda)."""
    start = time.perf_counter()
    X = np.load(f"{out_dir}/all_train_X.npy", mmap_mode="r")
    y = np.load(f"{out_dir}/all_train_y.npy", mmap_mode="r")
    Xv = np.load(f"{out_dir}/all_val_X.npy", mmap_mode="r")
    yv = np.load(f"{out_dir}/all_val_y.npy", mmap_mode="r")

    w, b, prev, used = np.zeros(X.shape[1]), 0.0, np.inf, 0
    for used in range(1, iters + 1):
        p = np.clip(sigmoid(X @ w + b), 1e-9, 1 - 1e-9)                # 1 predict
        loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))       # 2 loss
        g = X.T @ (p - y) / len(y)                                     # 3 gradients
        w, b = w - alpha * (g + lam * w), b - alpha * np.mean(p - y)   #   L2 on w only
        if abs(prev - loss) < tol:                                     # 4 converged
            break
        prev = loss
    val_pred = (sigmoid(Xv @ w + b) >= 0.5).astype(float)
    return {"alpha": alpha, "lam": lam, "w": w, "b": b, "iters": used,
            "val_f1": f1_score(yv, val_pred), "seconds": time.perf_counter() - start}
