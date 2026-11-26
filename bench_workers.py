"""Worker functions for Task 2: sequential vs parallel benchmark (rubric criterion 2).

Kept separate from workers.py so that a worker process only imports NumPy.
workers.py imports scikit-learn at the top, which makes every new process slower
to start; ping_heavy() below measures exactly how much slower.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"        # one maths thread per process, same as workers.py
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
import numpy as np
from threadpoolctl import threadpool_limits

# The environment variables above only work if NumPy has not been loaded yet. In a
# Jupyter kernel (PyCharm in particular) NumPy may already be loaded with one BLAS
# thread per core, and a multi-threaded X.T @ err adds its partial sums in a different
# order, so its last bits differ from a single-threaded worker. This call pins the
# already-loaded BLAS/OpenMP libraries to one thread as well.
threadpool_limits(limits=1)


def sigmoid(z):
    """Identical to workers.sigmoid (copied so this module does not import scikit-learn)."""
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def ping(hold):
    """Start-up probe: occupy a worker for `hold` seconds and report its process id."""
    time.sleep(hold)
    return os.getpid()


def ping_heavy(hold):
    """Same probe, but the worker also imports workers.py (and so scikit-learn)."""
    import workers  # noqa: F401  (the import itself is what we are timing)
    time.sleep(hold)
    return os.getpid()


def shard_gradient_timed(shard_id, out_dir, w, b):
    """Same maths as workers.shard_gradient, plus timings of each phase.

    Returns ((G_w, G_b, loss_sum, n_k), timing) where timing records which process
    ran the shard and how long it spent reading the file (I/O) and computing.
    """
    t0 = time.perf_counter()
    X = np.load(f"{out_dir}/s{shard_id:02d}_train_X.npy")
    y = np.load(f"{out_dir}/s{shard_id:02d}_train_y.npy")
    t1 = time.perf_counter()
    p = sigmoid(X @ w + b)
    err = p - y                                   # p_i - y_i
    p = np.clip(p, 1e-9, 1 - 1e-9)
    loss_sum = -(y * np.log(p) + (1 - y) * np.log(1 - p)).sum()
    g_w, g_b = X.T @ err, err.sum()
    t2 = time.perf_counter()
    timing = {"shard": shard_id, "pid": os.getpid(), "io_s": t1 - t0, "compute_s": t2 - t1}
    return (g_w, g_b, loss_sum, len(y)), timing
