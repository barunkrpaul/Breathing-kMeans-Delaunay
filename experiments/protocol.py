"""Full protocol for the manuscript. Resumable: finished (dataset,n,k,seed,config) rows are skipped.
Every solver is exact, so all configurations of one (dataset,n,k,seed) follow one trajectory;
the final centroids and SSE are compared with the first configuration run (Lloyd, full breathe)."""
import sys, os, time, math, numpy as np, pandas as pd, warnings
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
DATA = os.path.join(ROOT, "data")
from scipy.spatial.distance import cdist
from dnak_kmeans import run_algo, warmup
from dnak_kmeans.runner import delaunay_csr
from sklearn.cluster import kmeans_plusplus
warnings.filterwarnings("ignore")
OUT = os.path.join(RES, "protocol_results.csv")

def err_util_full(X, C):
    D = cdist(X, C, "sqeuclidean"); srt = D.argpartition(kth=1)[:, :2]; r = np.arange(len(X))
    da, db = D[r, srt[:, 0]], D[r, srt[:, 1]]
    a = np.where(da <= db, srt[:, 0], srt[:, 1]); d1, d2 = np.minimum(da, db), np.maximum(da, db)
    k = len(C); return np.bincount(a, d1, k), np.bincount(a, d2 - d1, k)

def err_util_local(X, C, a):
    k = len(C); ptr, idx = delaunay_csr(C)
    d1 = ((X - C[a]) ** 2).sum(1); d2 = np.full(len(X), np.inf)
    order = np.argsort(a, kind="stable"); b = np.searchsorted(a[order], np.arange(k + 1))
    for c in range(k):
        m = order[b[c]:b[c + 1]]
        if len(m): d2[m] = cdist(X[m], C[idx[ptr[c]:ptr[c + 1]]], "sqeuclidean").min(1)
    return np.bincount(a, d1, k), np.bincount(a, d2 - d1, k)

def breathe_in(C, E, n, m, rng):
    mx = (-E).argpartition(kth=m - 1)[:m]; rmse = math.sqrt(E.sum() / n)
    return np.concatenate([C, C[mx] + 0.01 * rmse * (rng.random((m, C.shape[1])) - 0.5)])

def breathe_out(C, U, m):
    cd = cdist(C, C, "sqeuclidean"); nn = cd.argpartition(kth=1)[:, 1]; Dm, Fr = set(), set()
    for u in U.argsort(kind="stable"):
        if u not in Fr:
            Dm.add(u)
            if len(Fr) + m < len(C): Fr.add(nn[u])
            if len(Dm) == m: break
    return C[sorted(set(range(len(C))) - Dm)]

def breathing(X, C0, solver, local, m=5, tol=1e-4, seed=0):
    rng = np.random.default_rng(seed); T = dict(lloyd=0.0, breathe=0.0, calls=0, iters=0, comps=0, cycles=0)
    def lloyd(C):
        r = run_algo(solver, X, C); T["lloyd"] += r["time"]; T["calls"] += 1; T["iters"] += r["iters"]; T["comps"] += r["comps"]; return r
    def eu(r):
        t = time.perf_counter(); o = err_util_local(X, r["C"], r["labels"]) if local else err_util_full(X, r["C"]); T["breathe"] += time.perf_counter() - t; return o
    t0 = time.perf_counter(); r = lloyd(C0); Eb, Cb = r["sse"], r["C"]; T["sse_init"] = r["sse"]; T["t_first"] = r["time"]
    while m > 0:
        T["cycles"] += 1
        E, _ = eu(r); r = lloyd(breathe_in(r["C"], E, len(X), m, rng))
        _, U = eu(r); r = lloyd(breathe_out(r["C"], U, m))
        if r["sse"] < Eb * (1 - tol): Eb, Cb = r["sse"], r["C"]
        else: m -= 1
    T["total"] = time.perf_counter() - t0; T["sse"] = Eb; T["C"] = Cb
    return T

_U = None
def data(name, n, rng):
    global _U
    if name == "uber":
        if _U is None:
            d = pd.read_csv(os.path.join(DATA, "uber-rides-data1.csv")); _U = d[(d.Lat.between(40.5, 41.0)) & (d.Lon.between(-74.3, -73.6))]
        s = _U.sample(n, random_state=0); X = np.c_[(s.Lon.values - s.Lon.mean()) * np.cos(np.deg2rad(s.Lat.mean())), s.Lat.values - s.Lat.mean()]
    else:
        q = pd.read_csv(os.path.join(DATA, "earthquakes-23k.csv")).dropna(); la, lo = np.deg2rad(q.Latitude.values), np.deg2rad(q.Longitude.values)
        X = np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]
    X = (X - X.mean(0)) / X.std()
    return np.ascontiguousarray(X + 1e-9 * np.random.default_rng(12345).standard_normal(X.shape))

CONFIGS = [("Lloyd", False), ("Hamerly", False), ("Exponion", False), ("Ball", False), ("DNAK-6", False),
           ("Hamerly", True), ("Exponion", True), ("Ball", True), ("DNAK-6", True)]

if __name__ == "__main__":
    warmup()
    done = set()
    if os.path.exists(OUT):
        p = pd.read_csv(OUT); done = set(zip(p.dataset, p.n, p.k, p.seed, p.solver, p.local))
    for spec in sys.argv[1].split(","):
        name, n, k, seeds = spec.split(":"); n, k = int(n), int(k)
        X = data(name, n, None); n = len(X)
        for seed in [int(s) for s in seeds.split("+")]:
            C0 = kmeans_plusplus(X, k, random_state=seed)[0]; ref = None
            for solver, local in CONFIGS:
                if (name, n, k, seed, solver, local) in done: continue
                T = breathing(X, C0, solver, local, seed=seed)
                if ref is None:
                    ref = T if (solver, local) == ("Lloyd", False) else None
                    refC, refS = T["C"], T["sse"]
                same = bool(T["C"].shape == refC.shape and np.allclose(T["C"], refC, rtol=0, atol=1e-9)) and abs(T["sse"] - refS) <= 1e-9 * refS
                row = dict(dataset=name, n=n, d=X.shape[1], k=k, seed=seed, solver=solver, local=local, total=T["total"], lloyd=T["lloyd"], breathe=T["breathe"],
                           t_first=T["t_first"], calls=T["calls"], cycles=T["cycles"], iters=T["iters"], comps=T["comps"], sse_init=T["sse_init"], sse=T["sse"], identical=same)
                pd.DataFrame([row]).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
                print(name, n, k, seed, solver, local, round(T["total"], 2), same, flush=True)
