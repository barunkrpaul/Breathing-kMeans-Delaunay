"""Builds every table, figure and number of the manuscript from the raw CSV files."""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from protocol import data, RES
from dnak_kmeans import run_algo, warmup
from dnak_kmeans.runner import delaunay_csr
from sklearn.cluster import kmeans_plusplus
from scipy.spatial import Voronoi, voronoi_plot_2d
T = RES + "/tables/"; FIG = RES + "/figures/"
d = pd.read_csv(RES + "/protocol_results.csv"); assert len(d) == 162 and d.identical.all()
d["cfg"] = d.solver.str.replace("-6", "") + np.where(d.local, "+L", "")
ORDER = [("quake", 23412, 64), ("quake", 23412, 256), ("uber", 50000, 64), ("uber", 50000, 256), ("uber", 200000, 256), ("uber", 50000, 1024)]
NAME = {"quake": "Earthquakes", "uber": "Uber"}
m = d.groupby(["dataset", "n", "k", "cfg"]).total.mean().unstack()
lo = d.groupby(["dataset", "n", "k", "cfg"]).total.min().unstack(); hi = d.groupby(["dataset", "n", "k", "cfg"]).total.max().unstack()
F = ["Lloyd", "Hamerly", "Exponion", "Ball", "DNAK"]; L = ["Hamerly+L", "Exponion+L", "Ball+L", "DNAK+L"]
def sg(v):
    t = f"{v:+.2f}"; return "0.00" if t in ("+0.00", "-0.00") else t
def f(x): return f"{x:.2f}" if x < 100 else f"{x:.1f}"
def lab(c): return f"{NAME[c[0]]} & {c[1]:,}".replace(",", "{,}") + f" & {c[2]}"
mac = {}
# ---- Table: total time
rows = []
for c in ORDER:
    best = min(m.loc[c, x] for x in F + L)
    cells = [(r"\textbf{%s}" % f(m.loc[c, x])) if m.loc[c, x] == best else f(m.loc[c, x]) for x in F + L]
    rows.append(lab(c) + " & " + " & ".join(cells) + r" \\")
open(T + "tab_time.tex", "w").write("\n".join(rows) + "\n")
# ---- Table: speed-up factors and breakdown
rows = []; S1, S2, S3, SH = [], [], [], []
bd = d.groupby(["dataset", "n", "k", "cfg"])[["lloyd", "breathe", "total"]].mean()
for c in ORDER:
    a = m.loc[c, "DNAK+L"]; s1 = m.loc[c, "Lloyd"] / a; s2 = min(m.loc[c, x] for x in F[1:4]) / a; s3 = min(m.loc[c, x] for x in L[:3]) / a
    sh = 100 * bd.loc[c + ("DNAK",), "breathe"] / bd.loc[c + ("DNAK",), "total"]; sh2 = 100 * bd.loc[c + ("DNAK+L",), "breathe"] / bd.loc[c + ("DNAK+L",), "total"]
    S1.append(s1); S2.append(s2); S3.append(s3); SH.append(sh)
    rows.append(lab(c) + f" & {bd.loc[c + ('DNAK',), 'lloyd']:.2f} & {bd.loc[c + ('DNAK',), 'breathe']:.2f} & {sh:.0f} & {bd.loc[c + ('DNAK+L',), 'breathe']:.2f} & {sh2:.0f} & {s1:.0f} & {s2:.1f} & {s3:.1f}" + r" \\")
open(T + "tab_speed.tex", "w").write("\n".join(rows) + "\n")
mac.update(SoneMin=f"{min(S1):.0f}", SoneMax=f"{max(S1):.0f}", StwoMin=f"{min(S2):.1f}", StwoMax=f"{max(S2):.1f}", SthreeMin=f"{min(S3):.1f}", SthreeMax=f"{max(S3):.1f}",
           ShareMin=f"{min(SH):.0f}", ShareMax=f"{max(SH):.0f}")
# share of the breathe step for the other accelerated solvers (full breathe)
oth = [100 * bd.loc[c + (x,), "breathe"] / bd.loc[c + (x,), "total"] for c in ORDER for x in ("Hamerly", "Exponion", "Ball")]
mac.update(OthShareMin=f"{min(oth):.0f}", OthShareMax=f"{max(oth):.0f}")
ll = [100 * bd.loc[c + ("Lloyd",), "breathe"] / bd.loc[c + ("Lloyd",), "total"] for c in ORDER]; mac.update(LloydShareMin=f"{min(ll):.0f}", LloydShareMax=f"{max(ll):.0f}")
# ---- run statistics and quality
s = d[d.cfg == "DNAK+L"].copy(); s["imp"] = 100 * (s.sse_init - s.sse) / s.sse_init
g = s.groupby(["dataset", "n", "k"]); rows = []
for c in ORDER:
    x = g.get_group(c); rows.append(lab(c) + f" & {x.cycles.mean():.1f} & {x.calls.mean():.1f} & {x.iters.mean():.0f} & {x.imp.mean():.2f} & {x.imp.min():.2f}--{x.imp.max():.2f}" + r" \\")
open(T + "tab_runs.tex", "w").write("\n".join(rows) + "\n")
mac.update(ImpMin=f"{s.imp.min():.1f}", ImpMax=f"{s.imp.max():.1f}", CallsMin=f"{s.calls.min()}", CallsMax=f"{s.calls.max()}", NRuns=str(len(d)), NTraj=str(len(s)))
# ---- package
p = pd.read_csv(RES + "/pkg_results.csv"); assert len(p) == 12
p["r1"] = p.pkg_1thread / p.ours; p["rd"] = p.pkg_default / p.ours; p["ds"] = 100 * (p.ours_sse - p.pkg_sse) / p.pkg_sse
rows = []
for c in [("quake", 23412, 256), ("uber", 50000, 256), ("uber", 200000, 256), ("uber", 50000, 1024)]:
    x = p[(p.dataset == c[0]) & (p.n == c[1]) & (p.k == c[2])]
    rows.append(lab(c) + f" & {f(x.pkg_1thread.mean())} & {f(x.pkg_default.mean())} & {x.ours.mean():.2f} & {x.pkg_1thread.mean() / x.ours.mean():.1f} & {x.pkg_default.mean() / x.ours.mean():.1f} & ${sg(x.ds.min())}$ to ${sg(x.ds.max())}$" + r" \\")
open(T + "tab_pkg.tex", "w").write("\n".join(rows) + "\n")
q = p.groupby(["dataset", "n", "k"])[["pkg_1thread", "pkg_default", "ours"]].mean()
mac.update(PkgOneMin=f"{(q.pkg_1thread / q.ours).min():.1f}", PkgOneMax=f"{(q.pkg_1thread / q.ours).max():.1f}", PkgDefMin=f"{(q.pkg_default / q.ours).min():.1f}", PkgDefMax=f"{(q.pkg_default / q.ours).max():.1f}",
           PkgDsMin=f"{p.ds.min():.2f}".replace("-", "$-$"), PkgDsMax=f"{p.ds.max():.2f}", PkgDsAbs=f"{p.ds.abs().max():.2f}")
# ---- random swap
r = pd.read_csv(RES + "/rswap_results.csv"); r = r[r.k == 256]; assert len(r) == 15 and r.identical.all()
a = r.groupby("solver").total.agg(["mean", "min", "max"]); r["imp"] = 100 * (r.sse_init - r.sse) / r.sse_init
open(T + "tab_rswap.tex", "w").write("\n".join(f"{x.replace('-6', '')} & {a.loc[x, 'mean']:.2f} & {a.loc[x, 'min']:.2f}--{a.loc[x, 'max']:.2f} & {a.loc[x, 'mean'] / a.loc['DNAK-6', 'mean']:.1f}" + r" \\" for x in ["Lloyd", "Hamerly", "Exponion", "Ball", "DNAK-6"]) + "\n")
oth = a.drop("DNAK-6")["mean"]
mac.update(RsBest=f"{oth.min() / a.loc['DNAK-6', 'mean']:.1f}", RsLloyd=f"{a.loc['Lloyd', 'mean'] / a.loc['DNAK-6', 'mean']:.0f}", RsAccMin=str(r.accepted.min()), RsAccMax=str(r.accepted.max()), RsImpMin=f"{r.imp.min():.1f}", RsImpMax=f"{r.imp.max():.1f}")
# ---- Delaunay degree at a converged solution
warmup(); deg = {}
for nm, n, k in [("uber", 50000, 256), ("quake", 0, 256)]:
    X = data(nm, n, None); C = run_algo("DNAK-6", X, kmeans_plusplus(X, k, random_state=0)[0])["C"]; ptr, idx = delaunay_csr(C); deg[nm] = np.diff(ptr).mean()
    if nm == "uber": XU, CU, PTR, IDX = X, C, ptr, idx
mac.update(DegTwo=f"{deg['uber']:.1f}", DegThree=f"{deg['quake']:.1f}")
open(T + "numbers.tex", "w").write("".join(f"\\newcommand{{\\{a}}}{{{b}}}\n" for a, b in mac.items()))
print(mac)
# ---- Figure 1: time per configuration
plt.rcParams.update({"font.size": 8, "font.family": "serif"})
fig, ax = plt.subplots(figsize=(4.8, 2.5)); cf = ["Lloyd", "Hamerly", "Exponion", "Ball", "DNAK", "DNAK+L"]; lb = ["Lloyd", "Hamerly", "Exponion", "Ball", "DNAK", "DNAK + local breathe"]
col = ["#d9d9d9", "#bdbdbd", "#969696", "#737373", "#6baed6", "#08519c"]; w = 0.14
for i, c in enumerate(cf):
    v = [m.loc[o, c] for o in ORDER]; e = [[m.loc[o, c] - lo.loc[o, c] for o in ORDER], [hi.loc[o, c] - m.loc[o, c] for o in ORDER]]
    ax.bar(np.arange(6) + (i - 2.5) * w, v, w, color=col[i], label=lb[i], yerr=e, error_kw=dict(lw=0.5, capsize=1))
ax.set_yscale("log"); ax.set_ylabel("Total time (s)"); ax.set_xticks(range(6))
ax.set_xticklabels([("Earthq." if o[0] == "quake" else "Uber") + f"\n$n$={o[1] // 1000}k\n$k$={o[2]}" for o in ORDER], fontsize=6.5)
ax.legend(ncol=3, fontsize=6.5, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.22)); ax.spines[["top", "right"]].set_visible(False)
fig.savefig(FIG + "time.pdf", bbox_inches="tight"); plt.close(fig)
# ---- Figure 2: the neighbours used by the local breathe step
fig, ax = plt.subplots(figsize=(3.0, 2.6)); V = Voronoi(CU); voronoi_plot_2d(V, ax=ax, show_points=False, show_vertices=False, line_width=0.4, line_colors="#888888")
cc = CU.mean(0); c = int(((CU - cc) ** 2).sum(1).argmin()); nb = IDX[PTR[c]:PTR[c + 1]]
lab_ = run_algo("DNAK-6", XU, CU, max_iter=1)["labels"]; P = XU[lab_ == c]
r_ = 2.2 * np.sqrt(((CU[nb] - CU[c]) ** 2).sum(1)).max()
ax.scatter(P[:, 0], P[:, 1], s=1, color="#9ecae1", zorder=1); ax.scatter(CU[:, 0], CU[:, 1], s=6, color="#bbbbbb", zorder=2)
for j in nb: ax.plot([CU[c, 0], CU[j, 0]], [CU[c, 1], CU[j, 1]], lw=0.7, color="#08519c", zorder=3)
ax.scatter(CU[nb, 0], CU[nb, 1], s=14, color="#08519c", zorder=4); ax.scatter(CU[c, 0], CU[c, 1], s=22, color="#cb181d", marker="s", zorder=5)
ax.set_xlim(CU[c, 0] - r_, CU[c, 0] + r_); ax.set_ylim(CU[c, 1] - r_, CU[c, 1] + r_); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
fig.savefig(FIG + "local.pdf", bbox_inches="tight"); plt.close(fig); print("neighbours of shown cell:", len(nb), "points:", len(P))
