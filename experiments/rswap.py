"""Second refinement method: random swap (Franti 2018). One centroid is moved to a random
data point, two k-means iterations follow, the swap is kept if the SSE drops.
The inner k-means is run by different exact solvers; the trial sequence is the same."""
import sys, os, time, numpy as np, pandas as pd, warnings
from protocol import data, RES
from dnak_kmeans import run_algo, warmup
from sklearn.cluster import kmeans_plusplus
warnings.filterwarnings("ignore"); warmup(); rows = []
TRIALS = 200
def rswap(X, C0, solver, seed):
    rng = np.random.default_rng(seed); t0 = time.perf_counter(); tk = 0.0
    r = run_algo(solver, X, C0); tk += r["time"]; C, E = r["C"], r["sse"]; E0 = E; acc = 0
    for _ in range(TRIALS):
        Ct = C.copy(); Ct[rng.integers(len(C))] = X[rng.integers(len(X))]
        r = run_algo(solver, X, Ct, max_iter=2); tk += r["time"]
        if r["sse"] < E: C, E = r["C"], r["sse"]; acc += 1
    return dict(total=time.perf_counter() - t0, kmeans=tk, sse_init=E0, sse=E, accepted=acc, C=C)
for name, n, k in [("uber", 50000, 256), ("uber", 50000, 1024)]:
    X = data(name, n, None)
    for seed in (0, 1, 2):
        C0 = kmeans_plusplus(X, k, random_state=seed)[0]; ref = None
        for s in ("Lloyd", "Hamerly", "Exponion", "Ball", "DNAK-6"):
            T = rswap(X, C0, s, seed)
            if ref is None: ref = T
            T["identical"] = bool(np.allclose(T["C"], ref["C"], rtol=0, atol=1e-9))
            row = dict(dataset=name, n=len(X), k=k, seed=seed, solver=s, **{a: b for a, b in T.items() if a != "C"})
            rows.append(row); print(row, flush=True); pd.DataFrame(rows).to_csv(RES + "/rswap_results.csv", index=False)
