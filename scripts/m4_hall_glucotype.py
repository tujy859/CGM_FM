"""M4: self-compute Hall glucotype labels (Hall et al. 2018, PLoS Biol).

Method follows the paper as closely as documented; unspecified details use
standard defaults (deviations marked DEV-*):

  - 2.5h windows (30 cells @5min), 75% overlap -> stride 7 cells = 35 min
    (DEV-1: paper's implied 37.5-min stride is not representable on the
    5-min grid; 7 cells is the closest)
  - linear imputation of gaps <15 min; windows with larger gaps excluded
  - Savitzky-Golay smoothing (win 7, poly 2) + per-window z-score
    (DEV-2: "polynomial smoothing" order unspecified -> SG default)
  - CID-DTW: symmetric2 step pattern, Sakoe-Chiba band 3 (~10% of window),
    complexity correction CE(x)=sqrt(sum diff^2), symmetrized
  - spectral clustering on kNN graph (smallest k with connected graph,
    binary symmetrized adjacency; DEV-3: weighting scheme unspecified),
    normalized Laplacian, k=3, k-means on eigenvector rows
  - clusters ordered by within-cluster mean window SD -> low/moderate/severe
  - subject glucotype = majority window class
  - DEV-4: pairwise matrix capped at 80 evenly-spaced windows/subject
    (paper used first-238/person; cap is for CPU tractability, unbiased
    for majority fractions)

Output: data/labels/glucotype_hall.json (+ prints cluster diagnostics vs
the paper's reported mean glucose per class: 77 / 96 / 122 mg/dL).

Run: code/CGM-JEPA/.venv/bin/python scripts/m4_hall_glucotype.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from numba import njit, prange
from scipy.signal import savgol_filter
from scipy.sparse.csgraph import connected_components
from scipy.linalg import eigh
from sklearn.cluster import KMeans

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "unified", "hall_2018.csv")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "labels", "glucotype_hall.json")

WIN = 30            # cells (2.5h)
STRIDE = 7          # cells (35 min, ~75% overlap)
MAX_GAP = 3         # cells (15 min): impute shorter, drop window if longer
SG_WIN, SG_POLY = 7, 2
CAP_WINDOWS = 80
K_CLUSTER = 3
SEED = 42


# ------------------------------------------------------------------ windows
def subject_windows(values, obs):
    """Valid 30-cell windows after <=15min linear imputation."""
    v = values.astype(np.float64).copy()
    obs = obs.astype(bool)
    # linear impute runs of <=3 missing cells
    i = 0
    while i < len(obs):
        if not obs[i]:
            j = i
            while j < len(obs) and not obs[j]:
                j += 1
            run = j - i
            if run <= MAX_GAP:
                left = v[i - 1] if i > 0 else None
                right = v[j] if j < len(v) else None
                if left is not None and right is not None:
                    for t in range(run):
                        v[i + t] = left + (right - left) * (t + 1) / (run + 1)
                    obs[i:j] = True
                elif left is not None:
                    v[i:j] = left
                    obs[i:j] = True
                elif right is not None:
                    v[i:j] = right
                    obs[i:j] = True
            i = j
        else:
            i += 1
    wins = []
    for s in range(0, len(v) - WIN + 1, STRIDE):
        w, wo = v[s:s + WIN], obs[s:s + WIN]
        if not wo.all() or np.isnan(w).any():
            continue
        wins.append(w)
    return wins


# ------------------------------------------------------------------ CID-DTW
@njit(cache=True, fastmath=True)
def _dtw_band(a, b, band):
    """Symmetric2 DTW with Sakoe-Chiba band. a, b are z-scored (WIN,)."""
    n = len(a)
    INF = 1e18
    D = np.full((n + 1, n + 1), INF)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        lo = max(1, i - band)
        hi = min(n, i + band)
        for j in range(lo, hi + 1):
            c = abs(a[i - 1] - b[j - 1])
            d = min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
            D[i, j] = c + d
    return D[n, n]


@njit(cache=True, fastmath=True, parallel=True)
def _cid_dtw_matrix(X, band):
    """Symmetrized CID-DTW over rows of X -> (N, N)."""
    N = X.shape[0]
    n = X.shape[1]
    ce = np.zeros(N)
    for i in range(N):
        s = 0.0
        for t in range(1, n):
            s += (X[i, t] - X[i, t - 1]) ** 2
        ce[i] = np.sqrt(s)
    D = np.zeros((N, N))
    for i in prange(N):
        for j in range(i + 1, N):
            d = _dtw_band(X[i], X[j], band)
            d *= 0.5 * (ce[i] / ce[j] + ce[j] / ce[i])
            D[i, j] = d
            D[j, i] = d
    return D


# ------------------------------------------------------------------ main
def main():
    df = pd.read_csv(DATA, parse_dates=["timestamp"], low_memory=False)
    rng = np.random.default_rng(SEED)

    per_subject = {}
    for subj, g in df.groupby("subject", sort=False):
        g = g.sort_values("timestamp")
        ts = pd.to_datetime(g["timestamp"])
        grid = ts.dt.floor("5min")
        s = pd.Series(pd.to_numeric(g["glucose_value"]).values, index=grid)
        s = s[~s.index.duplicated(keep="first")].sort_index()
        full = pd.date_range(s.index.min(), s.index.max(), freq="5min")
        s = s.reindex(full)
        obs = (~s.isna()).to_numpy()
        vals = s.to_numpy(dtype=np.float64)
        wins = subject_windows(vals, obs)
        if not wins:
            per_subject[subj] = []
            continue
        if len(wins) > CAP_WINDOWS:
            idx = np.linspace(0, len(wins) - 1, CAP_WINDOWS).astype(int)
            wins = [wins[i] for i in idx]
        per_subject[subj] = wins

    subjects = sorted(per_subject)
    X, owner, raw_means, raw_sds = [], [], [], []
    for si, subj in enumerate(subjects):
        for w in per_subject[subj]:
            X.append(savgol_filter(w, SG_WIN, SG_POLY).astype(np.float64))
            owner.append(si)
            raw_means.append(w.mean())
            raw_sds.append(w.std())
    X = np.stack(X)
    owner = np.array(owner)
    raw_means = np.array(raw_means)
    raw_sds = np.array(raw_sds)
    # global z-score with training-set-wide mean/SD (paper's prediction
    # protocol: "precomputed mean and standard deviation from the training
    # windows"); per-window z-scoring would erase the amplitude/variability
    # signal the three glucotypes are defined on
    gmean, gstd = X.mean(), X.std()
    X = ((X - gmean) / gstd).astype(np.float32)
    print(f"[glucotype] global z: mean={gmean:.1f} std={gstd:.1f} mg/dL")
    print(f"[glucotype] {len(subjects)} subjects, {len(X)} windows "
          f"(min/med per subject: {min(len(v) for v in per_subject.values())}/"
          f"{int(np.median([len(v) for v in per_subject.values()]))})")

    # CID-DTW distance matrix
    D = _cid_dtw_matrix(X, band=max(3, int(round(0.1 * WIN))))
    print(f"[glucotype] distance matrix done: mean={D[D > 0].mean():.3f}")

    # kNN graph: smallest k with connected graph
    order = np.argsort(D, axis=1)
    for k in range(3, 40):
        A = np.zeros_like(D, dtype=bool)
        for i in range(len(A)):
            A[i, order[i, 1:k + 1]] = True
        A = A | A.T
        ncomp, _ = connected_components(A, directed=False)
        if ncomp == 1:
            break
    print(f"[glucotype] kNN graph connected at k={k}")

    # normalized spectral clustering (von Luxburg tutorial)
    W = A.astype(np.float64)
    deg = W.sum(1)
    Dm = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-12)))
    L = np.eye(len(W)) - Dm @ W @ Dm
    _, V = eigh(L)
    Emb = V[:, 1:1 + K_CLUSTER]
    km = KMeans(n_clusters=K_CLUSTER, n_init=20, random_state=SEED).fit(Emb)
    lab = km.labels_

    # order clusters by mean window SD (raw space) -> low/moderate/severe
    order_lab = np.argsort([raw_sds[lab == c].mean() for c in range(K_CLUSTER)])
    rank = {c: r for r, c in enumerate(order_lab)}
    lab = np.array([rank[c] for c in lab])
    names = ["low", "moderate", "severe"]
    for c in range(K_CLUSTER):
        m = lab == c
        print(f"[glucotype] cluster {names[c]}: {m.sum()} windows, "
              f"mean glucose {raw_means[m].mean():.0f} mg/dL "
              f"(paper: 77/96/122), mean SD {raw_sds[m].mean():.1f}")

    # majority label per subject
    out = {}
    for si, subj in enumerate(subjects):
        wl = lab[owner == si]
        if len(wl) == 0:
            out[subj] = dict(glucotype=None, fractions={}, n_windows=0,
                             note="no valid windows")
            continue
        frac = [float((wl == c).mean()) for c in range(K_CLUSTER)]
        gl = int(np.argmax(frac))
        out[subj] = dict(glucotype=names[gl],
                         glucotype_3class=gl,
                         glucotype_severe=int(gl == 2),
                         fractions=dict(zip(names, [round(f, 3) for f in frac])),
                         n_windows=int(len(wl)))
    from collections import Counter
    counts = Counter(v["glucotype"] for v in out.values())
    print(f"[glucotype] subject distribution: {dict(counts)} "
          f"(paper: L 20 / M 14 / S 23)")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(dict(method="Hall 2018 self-computed (see scripts/m4_hall_glucotype.py docstring)",
                       seed=SEED, window_cells=WIN, stride_cells=STRIDE,
                       cap_windows=CAP_WINDOWS, subjects=out), f, indent=2)
    print(f"[glucotype] saved {OUT}")


if __name__ == "__main__":
    main()
