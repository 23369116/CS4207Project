"""Functions that run inside the worker processes."""
import os
os.environ["OMP_NUM_THREADS"] = "1"        # one maths thread per process, so workers do not fight over cores
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

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
    t["pickup_hour"] = t.tpep_pickup_datetime.dt.hour
    t = t.dropna(subset=features)
    X = t[features].to_numpy(dtype=float)
    y = (t.fare_amount > threshold).to_numpy(dtype=float)

    # Seed depends on the shard id, so every run gives the same split
    order = np.random.default_rng(seed + shard_id).permutation(len(y))
    order = order[:int(sample_frac * len(y))]      # use a fixed random sample of the rows
    n_train, n_val = int(0.70 * len(order)), int(0.15 * len(order))
    # Time-based train/test split
    parts = {"train": order[:n_train],
             "val": order[n_train:n_train + n_val],
             "test": order[n_train + n_val:]}
    for split, idx in parts.items():
        np.save(f"{out_dir}/s{shard_id:02d}_{split}_rawX.npy", X[idx])
        np.save(f"{out_dir}/s{shard_id:02d}_{split}_y.npy", y[idx])

    Xtr = X[parts["train"]]
    return {"n": len(Xtr), "sum": Xtr.sum(axis=0), "sumsq": (Xtr ** 2).sum(axis=0),
            "rows_kept": len(order)}
