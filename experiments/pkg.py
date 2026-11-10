"""Reference comparison with the public bkmeans package (scikit-learn Lloyd inside).
The package is timed with one thread and with its default thread count."""
import sys, os, time, numpy as np, pandas as pd, warnings
from threadpoolctl import threadpool_limits
from protocol import breathing, data, RES
from dnak_kmeans import warmup
from bkmeans import BKMeans
from sklearn.cluster import kmeans_plusplus
warnings.filterwarnings("ignore"); warmup(); rows = []
for name, n, k in [("quake", 0, 256), ("uber", 50000, 256), ("uber", 50000, 1024), ("uber", 200000, 256)]:
    X = data(name, n, None)
    for seed in (0, 1, 2):
        C0 = kmeans_plusplus(X, k, random_state=seed)[0]
        with threadpool_limits(1):
            t = time.perf_counter(); b = BKMeans(n_clusters=k, init=C0, n_init=1, random_state=seed).fit(X); t1 = time.perf_counter() - t
        t = time.perf_counter(); b2 = BKMeans(n_clusters=k, init=C0, n_init=1, random_state=seed).fit(X); t2 = time.perf_counter() - t
        T = breathing(X, C0, "DNAK-6", True, seed=seed)
        r = dict(dataset=name, n=len(X), k=k, seed=seed, pkg_1thread=t1, pkg_default=t2, pkg_sse=b.inertia_, ours=T["total"], ours_sse=T["sse"])
        rows.append(r); print(r, flush=True); pd.DataFrame(rows).to_csv(RES + "/pkg_results.csv", index=False)
