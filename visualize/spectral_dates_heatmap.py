#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spectral_dates_heatmap.py
--------------------------------
End-to-end pipeline:
1) Language->[years]  →  aligned histograms
2) Pairwise 1-D Wasserstein distances
3) Similarity (Gaussian kernel) → normalized Laplacian
4) Spectral embedding → KMeans clustering
5) Cluster-ordered heatmap + CSV exports

USAGE (dummy data):
    python spectral_dates_heatmap.py

USAGE (with your data as JSON mapping language->list):
    python spectral_dates_heatmap.py --json path/to/data.json --k 4 --bin-width 5

USAGE (with your data as CSV with columns Language,Year):
    python spectral_dates_heatmap.py --csv path/to/data.csv --k 4 --bin-width 5

Dependencies:
    pip install numpy pandas scikit-learn matplotlib
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import os
import sys


# -----------------------------
# Data loading
# -----------------------------
def load_data_from_json(path: str) -> Dict[str, List[int]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    data: Dict[str, List[int]] = {}
    for lang, years in raw.items():
        data[str(lang)] = [int(y) for y in years if y is not None]
    return data


def load_data_from_csv(path: str) -> Dict[str, List[int]]:
    """
    Expects a tall CSV with at least columns: Language, Year
    Multiple rows per language; each row is a single year mention.
    """
    df = pd.read_csv(path)
    if not {"Language", "Year"} <= set(df.columns):
        raise ValueError("CSV must contain columns: Language, Year")
    data: Dict[str, List[int]] = {}
    for lang, sub in df.groupby("Language"):
        data[str(lang)] = [int(y) for y in sub["Year"].dropna().astype(int).tolist()]
    return data


# -----------------------------
# Core math
# -----------------------------
def build_histograms(
    data: Dict[str, List[int]],
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    bin_width: int = 1,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Convert language->list-of-years into aligned probability histograms on a common grid.
    Returns (H, bin_edges, langs) where H[i] is the probability histogram for language i.
    """
    langs = list(data.keys())
    all_years = [y for ys in data.values() for y in ys]
    if len(all_years) == 0:
        raise ValueError("No year values found.")

    if year_min is None:
        year_min = min(all_years)
    if year_max is None:
        year_max = max(all_years)

    # Inclusive top by adding a bin step
    bin_edges = np.arange(int(year_min), int(year_max) + bin_width, bin_width, dtype=int)
    if bin_edges.size < 2:
        bin_edges = np.array([year_min, year_min + bin_width], dtype=int)

    H = []
    for lang in langs:
        years = np.asarray(data[lang], dtype=int)
        if years.size == 0:
            hist = np.zeros(len(bin_edges) - 1, dtype=float)
        else:
            hist, _ = np.histogram(years, bins=bin_edges)
        total = hist.sum()
        if total > 0:
            hist = hist.astype(float) / total
        H.append(hist)

    H = np.vstack(H)  # shape (n_langs, n_bins)
    return H, bin_edges, langs


def wasserstein_1d_from_hist(p: np.ndarray, q: np.ndarray, bin_width: float = 1.0) -> float:
    """
    Exact 1-D 1-Wasserstein (Earth Mover's) distance for equal-width bins:
        W1 = sum |CDF_p - CDF_q| * bin_width
    """
    if p.shape != q.shape:
        raise ValueError("Histogram shapes must match.")
    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)
    return float(np.sum(np.abs(cdf_p - cdf_q)) * bin_width)


def pairwise_wasserstein(H: np.ndarray, bin_width: float) -> np.ndarray:
    n = H.shape[0]
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = wasserstein_1d_from_hist(H[i], H[j], bin_width)
            D[i, j] = D[j, i] = d
    return D


def similarity_from_dist(D: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
    """
    Gaussian kernel: A_ij = exp( -D_ij^2 / (2*sigma^2) )
    If sigma is None, use the median of non-zero distances (robust default).
    """
    if sigma is None:
        vals = D[np.triu_indices_from(D, k=1)]
        nonzero = vals[vals > 0]
        sigma = float(np.median(nonzero)) if nonzero.size > 0 else 1.0
    if sigma <= 0:
        sigma = 1.0
    A = np.exp(-(D ** 2) / (2.0 * (sigma ** 2)))
    np.fill_diagonal(A, 0.0)
    return A


def normalized_graph_laplacian(A: np.ndarray) -> np.ndarray:
    """
    L_sym = I - D^{-1/2} A D^{-1/2}
    """
    d = A.sum(axis=1)
    with np.errstate(divide="ignore"):
        d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L = np.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt
    return L


def spectral_embed(L: np.ndarray, k: int) -> np.ndarray:
    """
    k smallest non-trivial eigenvectors of L (row-normalized).
    """
    evals, evecs = np.linalg.eigh(L)  # symmetric
    order = np.argsort(evals)
    evals = evals[order]
    evecs = evecs[:, order]
    tol = 1e-9
    nontrivial_idx = np.where(evals > tol)[0]
    if nontrivial_idx.size == 0:
        U = evecs[:, :k]
    else:
        start = nontrivial_idx[0]
        U = evecs[:, start : start + k]
        if U.shape[1] < k:
            extra = evecs[:, : (k - U.shape[1])]
            U = np.hstack([U, extra])
    U = normalize(U, norm="l2")
    return U


def spectral_clustering_pipeline(
    data: Dict[str, List[int]],
    k: int,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    bin_width: int = 1,
    sigma: Optional[float] = None,
):
    H, bins, langs = build_histograms(data, year_min, year_max, bin_width)
    D = pairwise_wasserstein(H, float(bin_width))
    A = similarity_from_dist(D, sigma)
    L = normalized_graph_laplacian(A)
    U = spectral_embed(L, k)
    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels = km.fit_predict(U)
    return {
        "languages": langs,
        "histograms": H,
        "bin_edges": bins,
        "distance_matrix": D,
        "similarity_matrix": A,
        "laplacian": L,
        "embedding": U,
        "labels": labels,
    }


# -----------------------------
# Plotting / export
# -----------------------------
def make_clustered_heatmap(
    res: dict,
    bin_width: int,
    title: str = "Language Date Histograms (cluster-ordered)",
    out_png: Optional[str] = "clustered_heatmap.png",
    out_order_csv: Optional[str] = "clustered_languages_order.csv",
    out_matrix_csv: Optional[str] = "clustered_language_heatmap_matrix.csv",
    row_rescale: bool = True,
):
    langs = res["languages"]
    labels = res["labels"]
    H = res["histograms"]
    bins = res["bin_edges"]
    U = res["embedding"]

    # Order rows by (cluster label, first embedding coordinate) for stability
    order = np.lexsort((U[:, 0], labels))
    H_ord = H[order, :]
    langs_ord = [f"{langs[i]} (Cluster {int(labels[i])})" for i in order]

    # Optionally rescale each row to its max for visual contrast
    H_disp = H_ord.copy()
    if row_rescale:
        row_max = np.maximum(H_ord.max(axis=1, keepdims=True), 1e-12)
        H_disp = H_ord / row_max

    # X tick labels (year centers)
    year_centers = (bins[:-1] + bins[1:]) // 2
    ncols = H_disp.shape[1]
    step = max(1, ncols // 12)
    xticks = np.arange(0, ncols, step)
    xticklabels = [str(int(year_centers[i])) for i in xticks]

    # Plot
    plt.figure(figsize=(12, 5 + 0.35 * len(langs_ord)))
    plt.imshow(H_disp, aspect="auto", interpolation="nearest")
    plt.yticks(np.arange(len(langs_ord)), langs_ord)
    plt.xticks(xticks, xticklabels, rotation=45, ha="right")
    plt.xlabel("Year (bin centers, width = {} years)".format(bin_width))
    plt.title(title)
    plt.colorbar(label="Relative frequency" + (" (row-normalized)" if row_rescale else ""))
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=200)
        print(f"[saved] {out_png}")
    plt.show()

    # Exports
    if out_order_csv:
        pd.DataFrame(
            {"Language": [langs[i] for i in order], "Cluster": [int(labels[i]) for i in order]}
        ).to_csv(out_order_csv, index=False)
        print(f"[saved] {out_order_csv}")

    if out_matrix_csv:
        pd.DataFrame(H_disp, index=[langs[i] for i in order], columns=[int(x) for x in year_centers]).to_csv(
            out_matrix_csv
        )
        print(f"[saved] {out_matrix_csv}")


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Spectral clustering + heatmap for language date lists.")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--json", type=str, help="Path to JSON file: {language: [year, ...], ...}")
    src.add_argument("--csv", type=str, help="Path to CSV with columns: Language, Year")
    ap.add_argument("--k", type=int, default=3, help="Number of clusters (default: 3)")
    ap.add_argument("--bin-width", type=int, default=5, help="Histogram bin width in years (default: 5)")
    ap.add_argument("--year-min", type=int, default=None, help="Minimum year (default: infer)")
    ap.add_argument("--year-max", type=int, default=None, help="Maximum year (default: infer)")
    ap.add_argument("--sigma", type=float, default=None, help="Kernel width for similarity (default: median distance)")
    ap.add_argument("--out-png", type=str, default="clustered_heatmap.png", help="Output heatmap PNG")
    ap.add_argument("--out-order-csv", type=str, default="clustered_languages_order.csv", help="Output CSV (order)")
    ap.add_argument("--out-matrix-csv", type=str, default="clustered_language_heatmap_matrix.csv", help="Output CSV (matrix)")
    ap.add_argument("--no-row-rescale", action="store_true", help="Disable row normalization for display")
    return ap.parse_args()


def main():
    args = parse_args()

    # Load data
    if args.json:
        data = load_data_from_json(args.json)
    elif args.csv:
        data = load_data_from_csv(args.csv)
    else:
        # ---- Dummy data (edit freely or supply --json/--csv) ----
        data = {
            "English":   [1898, 1899, 1900, 1902, 1905, 1910, 1911, 1912, 1914, 1918, 1950, 2000, 2001, 2005],
            "German":    [1871, 1914, 1918, 1933, 1939, 1945, 1961, 1989, 1990, 2000, 2005, 2010],
            "French":    [1789, 1790, 1791, 1870, 1871, 1914, 1916, 1918, 1940, 1945, 1968, 1969, 2000, 2001],
            "Spanish":   [1492, 1500, 1600, 1700, 1808, 1810, 1898, 1936, 1939, 1975, 1992, 2000, 2010, 2015],
            "Bulgarian": [1878, 1908, 1944, 1946, 1963, 1969, 1971, 1989, 1991, 2007, 2013, 2019],
            "Mongolian": [1206, 1279, 1368, 1911, 1921, 1990, 2000, 2005, 2010, 2015, 2020],
            "Swahili":   [1890, 1905, 1960, 1963, 1977, 1984, 1990, 1995, 2000, 2005, 2010],
        }
        print("[info] Using built-in dummy data. Provide --json or --csv to use your dataset.")

    # Run pipeline
    res = spectral_clustering_pipeline(
        data=data,
        k=args.k,
        year_min=args.year_min,
        year_max=args.year_max,
        bin_width=args.bin_width,
        sigma=args.sigma,
    )

    # Plot / save
    make_clustered_heatmap(
        res=res,
        bin_width=args.bin_width,
        title="Language Date Histograms (cluster-ordered)",
        out_png=args.out_png,
        out_order_csv=args.out_order_csv,
        out_matrix_csv=args.out_matrix_csv,
        row_rescale=not args.no_row_rescale,
    )


if __name__ == "__main__":
    main()
