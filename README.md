# Breathing k-means with Delaunay Neighbours

Code and raw results for the manuscript

> B. Kr. Paul and K. Giri, *Breathing k-means with Delaunay Neighbours: An Exact Speed-up for Low-Dimensional Data*.

Breathing k-means (Fritzke, 2020) improves a k-means solution by adding and
removing a few centroids in turns, with a run of Lloyd's algorithm after every
change. This repository makes it faster for data in two or three dimensions.
The output is not changed.

Two things are done.

1. Lloyd's algorithm is replaced by [DNAK](https://github.com/barunkrpaul/dnak-kmeans),
   an exact accelerated k-means that searches among Delaunay neighbours of the centroids.
2. The error and the utility of each centroid are computed from the cluster labels and
   from the Delaunay neighbours of each centroid. The full n x k distance matrix of the
   breathing step is no longer needed. This is exact, because the second nearest
   centroid of a point is always a Delaunay neighbour of its nearest centroid.

## Results in brief

Three k-means++ seedings per configuration, one thread, same framework for all solvers.
Total time in seconds (mean).

| Data | n | k | Lloyd | Best other exact solver | DNAK + local step |
|:--|--:|--:|--:|--:|--:|
| Earthquakes | 23,412 | 64 | 2.02 | 0.39 | 0.22 |
| Earthquakes | 23,412 | 256 | 9.12 | 2.21 | 0.70 |
| Uber | 50,000 | 64 | 9.56 | 1.33 | 0.51 |
| Uber | 50,000 | 256 | 28.04 | 5.47 | 0.68 |
| Uber | 200,000 | 256 | 109.9 | 14.21 | 1.84 |
| Uber | 50,000 | 1024 | 122.0 | 38.94 | 2.10 |

"Best other" is the fastest of Hamerly, Exponion and Ball k-means with the usual
full breathing step. All 162 runs returned the same centroids and the same SSE as
the Lloyd-based run. The raw rows are in `results/protocol_results.csv`.

## Layout

```
experiments/
  protocol.py       breathing k-means with a pluggable exact solver; main timing protocol
  pkg.py            comparison with the public bkmeans package
  rswap.py          random swap with the same solvers
  make_outputs.py   builds every table, figure and number of the paper from the CSV files
  get_data.sh       downloads the two data files
  run_all.sh        runs everything in order
results/
  protocol_results.csv   one row per (data, n, k, seeding, solver, breathing step)
  pkg_results.csv        bkmeans package against the proposed setting
  rswap_results.csv      random swap
  tables/, figures/      output of make_outputs.py, as used in the paper
data/                    filled by get_data.sh (not stored in the repository)
```

## Reproducing

```bash
git clone https://github.com/barunkrpaul/breathing-kmeans-delaunay.git
cd breathing-kmeans-delaunay
pip install -r requirements.txt
./experiments/get_data.sh
./experiments/run_all.sh
```

The main protocol takes about 50 minutes on one core. It is resumable: rows
already in `results/protocol_results.csv` are skipped. To rebuild only the tables
and figures from the stored results, run `python3 experiments/make_outputs.py`.

Times depend on the machine. The stored results were measured on a shared cloud
machine with two cores (Python 3.13, NumPy 2.5.3, SciPy 1.18.1, Numba 0.68.1,
scikit-learn 1.9.1, bkmeans 1.3, DNAK 1.1.0).

## Data

Both files come from <https://github.com/plotly/datasets>:
`uber-rides-data1.csv` (Uber pick-ups in New York City, April to June 2014) and
`earthquakes-23k.csv` (23,412 earthquake epicentres).

## Limits

The method needs the Delaunay triangulation of the centroids and is meant for
two or three dimensions. The comparison solvers are the Numba versions from the
DNAK repository, not the original C++ programs.

## Licence

MIT. See `LICENSE`.
