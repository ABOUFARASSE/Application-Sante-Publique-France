# ============================================================
# RGPH 2024 | Commune Clustering (Household Profiling)
# ------------------------------------------------------------
# INPUT  : communes_rgph_long.csv (one row = commune-year)
#          -> this script filters automatically to year==2024
# OUTPUT : ./outputs_2024/ (tables CSV + figures + labeled dataset)
#
# Goal (paper-grade workflow):
#   1) Data QA: schema checks + compositional sums checks + optional renormalization
#   2) Missing handling: distance to paved road -> hierarchical median imputation
#   3) Correct transforms:
#        - CLR for compositional blocks (shares summing to 100)
#        - Logit for non-compositional rates (comfort/service access)
#        - log1p(distance) and log(n_households) for skewed continuous vars
#   4) Robust scaling (median/IQR), PCA dimension reduction (keep ~90% variance)
#   5) Clustering:
#        - Main: k-medoids (PAM) in PCA space (robust to outliers)
#        - Robustness: k-means and HDBSCAN
#        - Stability: subsampling bootstrap with ARI/NMI
#   6) Outputs:
#        - Clean dataset (2024)
#        - Cluster labels per commune (QGIS-ready)
#        - Cluster profiles table (weighted/unweighted)
#        - Diagnostics tables + figures
#        - Design-robustness stratification (<2000 vs >=2000 households)
#
# Dependencies:
#   pip install pandas numpy scikit-learn matplotlib
#   pip install scikit-learn-extra   # for KMedoids (recommended)
#   pip install hdbscan              # for HDBSCAN robustness (recommended)
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd

# --- Core ML tools ---
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,  # agreement metric between two partitions (stability)
    calinski_harabasz_score,  # another internal clustering diagnostic (higher is better)
    normalized_mutual_info_score,  # agreement metric between two partitions (stability)
    silhouette_score,  # cluster separation diagnostic (higher is better)
)
from sklearn.cluster import KMeans

# --- Plotting (paper figures) ---
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")  # keep logs clean for batch runs; remove if debugging

# --- Optional: k-medoids (main estimator) ---
try:
    from sklearn_extra.cluster import KMedoids
    HAS_KMEDOIDS = True
except Exception:
    HAS_KMEDOIDS = False

# --- Optional: HDBSCAN (density-based robustness) ---
try:
    import hdbscan
    HAS_HDBSCAN = True
except Exception:
    HAS_HDBSCAN = False


# ============================================================
# 0) CONFIGURATION (change here only; rest is fully automatic)
# ============================================================

# ---- I/O ----
INPUT_PATH = "communes_rgph_long.csv"  # path to your long file
OUT_DIR = "outputs_2024"  # all outputs go here
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Analysis scope: 2024 only ----
YEAR_KEEP = 2024

# ---- Compositional QA settings ----
# Blocks of percentages (housing shares, cooking shares, etc.) should sum ~100.
# Due to rounding / residual categories, we allow a tolerance band; beyond that,
# we can optionally renormalize to sum to 100.
BLOCK_SUM_TOL = (98.0, 102.0)
RENORMALIZE_BLOCKS = True

# ---- Missing handling for accessibility distance ----
# HCP files sometimes have "..." or "…" placeholders -> treated as missing.
# If enabled, impute via median(province) -> median(region) -> median(national).
IMPUTE_DISTANCE = True
DIST_COL = "distance_pavedroad_km_mean"

# ---- Transformation parameters ----
EPS = 1e-4  # small constant to avoid log(0), logit(0/1), CLR zero issues
PCA_VARIANCE_TARGET = 0.90  # retain PCA components explaining ~90% of variance

# ---- Clustering range for selecting K ----
K_MIN, K_MAX = 3, 10
RANDOM_STATE = 42

# ---- Stability analysis (subsampling) ----
N_BOOT = 50
SUBSAMPLE_FRAC = 0.80

# ---- Cluster profiles: optionally weight by number of households ----
WEIGHT_COL = "n_households"
USE_WEIGHTS_IN_PROFILES = True

# ---- Robustness related to RGPH 2024 long questionnaire design ----
# Communes with >=2000 households typically have 20% long questionnaire sample,
# so proportions may be noisier. We stratify by this threshold.
DESIGN_THRESHOLD = 2000

# ---- Optional shrinkage (rare shares) ----
# If you observe unstable clusters driven by very small categories, you can enable
# Beta-binomial shrinkage using an "effective sample size" approximation.
ENABLE_SHRINKAGE = False
SHRINK_ALPHA = 0.5  # Jeffreys prior
SHRINK_BETA = 0.5


# ============================================================
# 1) EXPECTED SCHEMA (variables used for clustering)
# ============================================================

# Required ID columns. 'region' is optional (if absent, we'll substitute province for imputation).
KEY_COLS_REQUIRED = ["year", "code_commune", "commune_name", "province_prefecture", "n_households"]
KEY_COLS_OPTIONAL = ["region"]

# Continuous variables (not compositional)
CONT_COLS = [
    "hh_size_mean",
    "persons_per_room_mean",
    "distance_pavedroad_km_mean",
]

# Compositional blocks (shares in %, each block ~100)
BLOCKS_COMPOSITIONAL = {
    "housing": [
        "housing_villa_pct",
        "housing_apartment_pct",
        "housing_maison_marocaine_pct",
        "housing_sommaire_pct",
        "housing_rural_pct",
        "housing_other_pct",
    ],
    "tenure": [
        "tenure_owner_pct",
        "tenure_renter_pct",
        "tenure_other_pct",
    ],
    "housing_age": [
        "age_lt10_pct",
        "age_10_19_pct",
        "age_20_49_pct",
        "age_ge50_pct",
    ],
    "sanitation": [
        "sanit_public_sewer_pct",
        "sanit_septic_pct",
        "sanit_other_pct",
    ],
    "waste": [
        "waste_bin_pct",
        "waste_truck_pct",
        "waste_nature_pct",
        "waste_other_pct",
    ],
    # For 2024, cooking fuel block should be a proper composition (sum ~100)
    "cook": [
        "cook_gas_pct",
        "cook_elec_pct",
        "cook_charcoal_pct",
        "cook_wood_pct",
        "cook_other_pct",
    ],
}

# Non-compositional rates (percentages that do NOT sum to 100)
COMFORT_RATE_COLS = [
    "comfort_kitchen_pct",
    "comfort_wc_pct",
    "comfort_bathroom_pct",
    "comfort_electricity_pct",
    "comfort_running_water_pct",
]

# "Readable variables" for cluster profiles tables (output in original scale)
READABLE_VARS = (
    [
        "hh_size_mean",
        "persons_per_room_mean",
        "distance_pavedroad_km_mean",
        "n_households",
    ]
    + COMFORT_RATE_COLS
    + sum(BLOCKS_COMPOSITIONAL.values(), [])
)


# ============================================================
# 2) UTILITY FUNCTIONS (QA, transforms, models, outputs)
# ============================================================


def ensure_columns(df, cols, name="dataframe"):
    """Fail fast if expected columns are missing (prevents silent errors later)."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[ERROR] Missing columns in {name}: {missing}")


def coerce_numeric(df, cols):
    """Convert listed columns to numeric; invalid parsing becomes NaN."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def block_sum(df, cols):
    """Row-wise sum of a compositional block; NaN if any part is NaN (skipna=False)."""
    return df[cols].sum(axis=1, skipna=False)


def renormalize_block(df, cols, target_sum=100.0):
    """
    Renormalize a block so each row sums exactly to target_sum.
    Use when rounding or residual categories cause sums outside tolerance.
    """
    s = df[cols].sum(axis=1, skipna=False)
    out = df[cols].copy()
    mask = (s.notna()) & (s != 0)
    out.loc[mask, :] = out.loc[mask, :].div(s[mask], axis=0) * target_sum
    df.loc[:, cols] = out
    return df


def hierarchical_median_impute(df, value_col, groups):
    """
    Hierarchical imputation:
      fill missing by median within group1; if still missing -> group2; then national median.
    This is conservative and transparent; appropriate for accessibility proxies.
    """
    out = df.copy()
    for g in groups:
        if g in out.columns:
            med = out.groupby(g)[value_col].transform("median")
            out[value_col] = out[value_col].fillna(med)
    out[value_col] = out[value_col].fillna(out[value_col].median())
    return out


def logit_transform_pct(x, eps=1e-4):
    """
    Logit transform for a rate in percent.
    We clip to (eps, 1-eps) to avoid infinite values.
    """
    p = x / 100.0
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def clr_transform_block(df, cols, eps=1e-4):
    """
    Centered log-ratio (CLR) transform for a compositional block in percent.

    Why:
      - A composition lies on the simplex (parts sum to 1).
      - Euclidean distances on raw shares can produce artefacts.
      - CLR maps the simplex to real space where Euclidean geometry is coherent.

    Steps:
      1) convert % -> proportions
      2) reclose to sum=1 (in case of rounding)
      3) add eps and take log
      4) subtract mean log across parts (per row)
    """
    X = df[cols].copy()
    P = (X / 100.0).clip(lower=0)

    # Reclose rows to sum=1 (only where sum>0)
    row_sum = P.sum(axis=1, skipna=False)
    mask = (row_sum.notna()) & (row_sum > 0)
    P.loc[mask, :] = P.loc[mask, :].div(row_sum[mask], axis=0)

    # Avoid log(0)
    P = P.clip(lower=eps)
    logP = np.log(P)

    # CLR: log(p_j) - mean_k log(p_k)
    gm = logP.mean(axis=1)
    clr = logP.sub(gm, axis=0)
    clr.columns = [f"{c}__clr" for c in cols]
    return clr


def robust_scale(X):
    """
    Robust scaling: center by median, scale by IQR.
    Why:
      - census aggregates can have outliers (very large urban communes).
      - Robust scaling reduces outlier influence vs standard z-score.
    """
    scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))
    Xs = scaler.fit_transform(X)
    return Xs, scaler


def pca_reduce(Xs, variance_target=0.90, random_state=42):
    """
    PCA reduction:
      - fit PCA on standardized features
      - keep the smallest number of PCs that reach the target cumulative variance
    """
    pca_full = PCA(random_state=random_state)
    pca_full.fit(Xs)
    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp = int(np.searchsorted(cumsum, variance_target) + 1)

    pca = PCA(n_components=n_comp, random_state=random_state)
    Z = pca.fit_transform(Xs)
    return Z, pca, pca_full, n_comp, cumsum


def fit_kmedoids(Z, k, random_state=42):
    """
    Main clustering estimator: k-medoids (PAM) in PCA space.
    More robust than k-means because centers are observed points (medoids).
    """
    if not HAS_KMEDOIDS:
        raise RuntimeError("KMedoids not available. Install scikit-learn-extra.")
    model = KMedoids(n_clusters=k, metric="euclidean", init="k-medoids++", random_state=random_state)
    labels = model.fit_predict(Z)
    return model, labels


def fit_kmeans(Z, k, random_state=42):
    """Benchmark clustering estimator: k-means in PCA space."""
    model = KMeans(n_clusters=k, n_init=20, random_state=random_state)
    labels = model.fit_predict(Z)
    return model, labels


def fit_hdbscan(Z):
    """
    Robustness estimator: HDBSCAN.
    Useful when clusters have varying density and for identifying outliers (-1).
    """
    if not HAS_HDBSCAN:
        raise RuntimeError("HDBSCAN not available. Install hdbscan.")
    model = hdbscan.HDBSCAN(min_cluster_size=30, min_samples=10)
    labels = model.fit_predict(Z)
    return model, labels


def choose_k_by_metrics(Z, k_min=3, k_max=10):
    """
    Compute internal criteria over a range of K:
      - Silhouette (higher better)
      - Calinski–Harabasz (higher better)

    We compute metrics for:
      - k-means always
      - k-medoids if available (recommended main estimator)
    """
    rows = []
    for k in range(k_min, k_max + 1):
        # k-means
        _, lab_km = fit_kmeans(Z, k, RANDOM_STATE)
        rows.append(
            {
                "method": "kmeans",
                "k": k,
                "silhouette": silhouette_score(Z, lab_km),
                "calinski_harabasz": calinski_harabasz_score(Z, lab_km),
            }
        )
        # k-medoids (if installed)
        if HAS_KMEDOIDS:
            _, lab_kmd = fit_kmedoids(Z, k, RANDOM_STATE)
            rows.append(
                {
                    "method": "kmedoids",
                    "k": k,
                    "silhouette": silhouette_score(Z, lab_kmd),
                    "calinski_harabasz": calinski_harabasz_score(Z, lab_kmd),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_stability(Z, labels_ref, fit_fun, k, n_boot=50, frac=0.8, random_state=42):
    """
    Subsampling stability:
      - repeatedly draw a fraction of observations without replacement
      - refit clustering on the subsample
      - compare subsample labels to the reference labels restricted to subsample
    Metrics:
      - ARI and NMI (agreement)
    """
    rng = np.random.default_rng(random_state)
    n = Z.shape[0]
    stats = []
    for b in range(n_boot):
        idx = rng.choice(n, size=int(frac * n), replace=False)
        Zb = Z[idx]
        _, lab_b = fit_fun(Zb, k, random_state + b)
        lab_ref_b = labels_ref[idx]
        stats.append(
            {
                "boot": b,
                "n": len(idx),
                "ARI": adjusted_rand_score(lab_ref_b, lab_b),
                "NMI": normalized_mutual_info_score(lab_ref_b, lab_b),
            }
        )
    return pd.DataFrame(stats)


def weighted_cluster_profiles(df, label_col, vars_readable, weight_col=None):
    """
    Compute cluster profiles in ORIGINAL SCALE (percentages, means, etc.).
    Optionally weight by number of households (to reflect population relevance).
    """
    out = []
    for cl, g in df.groupby(label_col):
        row = {"cluster": cl, "n_rows": len(g)}
        if weight_col and weight_col in g.columns and g[weight_col].notna().any():
            w = g[weight_col].astype(float).fillna(0.0)
            wsum = w.sum()
            row["weight_sum"] = wsum
            for v in vars_readable:
                x = pd.to_numeric(g[v], errors="coerce")
                row[v] = (np.nansum(x * w) / wsum) if (wsum > 0) else np.nanmean(x)
        else:
            row["weight_sum"] = np.nan
            for v in vars_readable:
                row[v] = np.nanmean(pd.to_numeric(g[v], errors="coerce"))
        out.append(row)
    return pd.DataFrame(out).sort_values("cluster")


def beta_binomial_shrink_pct(pct, n_eff, alpha=0.5, beta=0.5):
    """
    Optional shrinkage for noisy small categories:
      - approximate counts y = n_eff * p
      - posterior mean under Beta(alpha,beta) prior
    Returns percentage.
    """
    p = np.clip(pct / 100.0, 0.0, 1.0)
    n_eff = np.clip(n_eff, 1.0, None)
    y = n_eff * p
    post = (y + alpha) / (n_eff + alpha + beta)
    return 100.0 * post


# ============================================================
# 3) LOAD DATA + FILTER TO 2024 + BASIC CLEANING
# ============================================================

# Load as CSV. Keep commune codes as strings to preserve leading zeros.
df = pd.read_csv(INPUT_PATH, dtype={"code_commune": str})

# Check required columns exist (fail fast if not)
ensure_columns(df, KEY_COLS_REQUIRED, "input")

# Ensure optional columns exist; if missing, create as NaN
for c in KEY_COLS_OPTIONAL:
    if c not in df.columns:
        df[c] = np.nan

# Normalize ID fields and types
df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
df["code_commune"] = df["code_commune"].astype(str).str.strip()
df["commune_name"] = df["commune_name"].astype(str).str.strip()
df["province_prefecture"] = df["province_prefecture"].astype(str).str.strip()

# Filter the dataset to year==2024 (this makes the paper cross-sectional)
df = df[df["year"] == YEAR_KEEP].copy()
if df.empty:
    raise ValueError(f"[ERROR] No observations for year=={YEAR_KEEP}. Check the 'year' column.")

# If region is missing, substitute province_prefecture for hierarchical imputation
df["region"] = df["region"].fillna(df["province_prefecture"])

# Check that all variables used by the pipeline are present
all_var_cols = CONT_COLS + COMFORT_RATE_COLS + sum(BLOCKS_COMPOSITIONAL.values(), [])
ensure_columns(df, all_var_cols, "input")

# Convert numeric columns (invalid parsing becomes NaN)
df = coerce_numeric(df, all_var_cols + ["n_households"])

# Replace non-numeric placeholders for distance (common in Excel exports)
df[DIST_COL] = pd.to_numeric(df[DIST_COL], errors="coerce")


# ============================================================
# 4) QA: CHECK COMPOSITIONAL SUMS + OPTIONAL RENORMALIZATION
# ============================================================

# For each compositional block, compute row sums and how often sums fall outside tolerance
qa_rows = []
for bname, cols in BLOCKS_COMPOSITIONAL.items():
    s = block_sum(df, cols)
    qa_rows.append(
        {
            "block": bname,
            "count_nonmissing": int(s.notna().sum()),
            "mean_sum": float(np.nanmean(s)),
            "min_sum": float(np.nanmin(s)),
            "median_sum": float(np.nanmedian(s)),
            "max_sum": float(np.nanmax(s)),
            "share_outside_tol": float(((s < BLOCK_SUM_TOL[0]) | (s > BLOCK_SUM_TOL[1])).mean()),
        }
    )
qa_blocks = pd.DataFrame(qa_rows)
qa_blocks.to_csv(os.path.join(OUT_DIR, "QA_block_sums_2024.csv"), index=False)

# If enabled, renormalize any row whose block sum is outside tolerance band
if RENORMALIZE_BLOCKS:
    for bname, cols in BLOCKS_COMPOSITIONAL.items():
        s = block_sum(df, cols)
        out_tol = (s.notna()) & ((s < BLOCK_SUM_TOL[0]) | (s > BLOCK_SUM_TOL[1]))
        if out_tol.any():
            df.loc[out_tol, :] = renormalize_block(df.loc[out_tol, :], cols)


# ============================================================
# 5) HANDLE MISSING DISTANCE (ACCESSIBILITY) BY HIERARCHICAL IMPUTATION
# ============================================================

if IMPUTE_DISTANCE:
    # Fill missing distance values using median at progressively broader levels:
    # province -> region -> national
    df = hierarchical_median_impute(df, DIST_COL, ["province_prefecture", "region"])


# ============================================================
# 6) OPTIONAL 2024 DESIGN ROBUSTNESS: SHRINKAGE ON RARE COOKING SHARES
# ============================================================

if ENABLE_SHRINKAGE:
    # Effective sample size approximation:
    #   if commune <2000 households -> assume full long questionnaire (n_eff = n_households)
    #   else -> assume 20% sample (n_eff = 0.2 * n_households)
    n_eff = np.where(
        df["n_households"].astype(float) < DESIGN_THRESHOLD,
        df["n_households"].astype(float),
        0.2 * df["n_households"].astype(float),
    )
    # Apply shrinkage only to cooking shares (typically rare categories)
    for c in BLOCKS_COMPOSITIONAL["cook"]:
        df[c] = beta_binomial_shrink_pct(df[c].astype(float), n_eff, alpha=SHRINK_ALPHA, beta=SHRINK_BETA)


# Save a clean 2024 dataset (for transparency / replication)
df.to_csv(os.path.join(OUT_DIR, "communes_rgph_2024_clean.csv"), index=False)


# ============================================================
# 7) FEATURE ENGINEERING (TRANSFORMS SUITABLE FOR CLUSTERING)
# ============================================================

d = df.copy()

# --- Continuous transforms ---
# log(n_households) captures size on a compressed scale (very skewed)
d["n_households_log"] = np.log(np.clip(d["n_households"].astype(float), 1, None))

# log1p(distance) stabilizes heavy tail in accessibility distances
d["distance_log1p"] = np.log1p(d[DIST_COL])

# --- Comfort rates: logit transform ---
# Avoids compressing differences near 0%/100% and yields a real-valued variable
for c in COMFORT_RATE_COLS:
    d[f"{c}__logit"] = logit_transform_pct(d[c].astype(float), eps=EPS)

# --- Compositional blocks: CLR transform for each block ---
# This is the critical step preventing spurious Euclidean geometry on shares.
clr_parts = []
for bname, cols in BLOCKS_COMPOSITIONAL.items():
    clr_parts.append(clr_transform_block(d, cols, eps=EPS))
X_clr = pd.concat(clr_parts, axis=1)

# --- Final feature matrix used for clustering ---
base_cont = ["hh_size_mean", "persons_per_room_mean", "n_households_log", "distance_log1p"]
X = pd.concat(
    [
        d[base_cont],
        d[[f"{c}__logit" for c in COMFORT_RATE_COLS]],
        X_clr,
    ],
    axis=1,
)

# Missing values in features: impute with global median (simple and reproducible)
# (We already imputed distance; this is a safety net.)
for c in X.columns:
    X[c] = X[c].fillna(X[c].median())


# ============================================================
# 8) ROBUST SCALING + PCA
# ============================================================

# Robust scaling (median/IQR) reduces the influence of large urban outliers
Xs, scaler = robust_scale(X)

# PCA reduces dimensionality and noise, keeping the most informative structure
Z, pca, pca_full, n_comp, cumsum = pca_reduce(
    Xs, variance_target=PCA_VARIANCE_TARGET, random_state=RANDOM_STATE
)

# Export PCA diagnostics (for the paper: variance explained + loadings)
pd.DataFrame(
    {
        "component": np.arange(1, len(pca_full.explained_variance_ratio_) + 1),
        "explained_variance_ratio": pca_full.explained_variance_ratio_,
        "cumulative_explained_variance": cumsum,
    }
).to_csv(os.path.join(OUT_DIR, "PCA_variance_2024.csv"), index=False)

pd.DataFrame(
    pca.components_.T,
    index=X.columns,
    columns=[f"PC{i+1}" for i in range(n_comp)],
).to_csv(os.path.join(OUT_DIR, "PCA_loadings_2024.csv"), index=True)

# Figure: cumulative variance curve
plt.figure()
plt.plot(np.arange(1, len(cumsum) + 1), cumsum, marker="o")
plt.axhline(PCA_VARIANCE_TARGET, linestyle="--")
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("PCA cumulative variance (2024)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "F_PCA_cumvar_2024.png"), dpi=200)
plt.close()


# ============================================================
# 9) SELECT NUMBER OF CLUSTERS K (INTERNAL CRITERIA)
# ============================================================

# Compute silhouette + Calinski–Harabasz for K in [K_MIN, K_MAX]
k_table = choose_k_by_metrics(Z, k_min=K_MIN, k_max=K_MAX)
k_table.to_csv(os.path.join(OUT_DIR, "K_selection_metrics_2024.csv"), index=False)

# Figures: silhouette vs K and CH vs K
plt.figure()
for method in k_table["method"].unique():
    sub = k_table[k_table["method"] == method].sort_values("k")
    plt.plot(sub["k"], sub["silhouette"], marker="o", label=method)
plt.xlabel("K")
plt.ylabel("Silhouette")
plt.title("K selection: silhouette (2024)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "F_silhouette_vs_K_2024.png"), dpi=200)
plt.close()

plt.figure()
for method in k_table["method"].unique():
    sub = k_table[k_table["method"] == method].sort_values("k")
    plt.plot(sub["k"], sub["calinski_harabasz"], marker="o", label=method)
plt.xlabel("K")
plt.ylabel("Calinski–Harabasz")
plt.title("K selection: Calinski–Harabasz (2024)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "F_CH_vs_K_2024.png"), dpi=200)
plt.close()

# Automatic (conservative) K* selection:
# Prefer k-medoids if available; otherwise fall back to k-means.
if HAS_KMEDOIDS and (k_table["method"].eq("kmedoids").any()):
    best = k_table[k_table["method"] == "kmedoids"].sort_values("silhouette", ascending=False).iloc[0]
else:
    best = k_table[k_table["method"] == "kmeans"].sort_values("silhouette", ascending=False).iloc[0]
K_STAR = int(best["k"])
BEST_METHOD = str(best["method"])
print(f"[INFO] Selected K*: {K_STAR} (method={BEST_METHOD})")


# ============================================================
# 10) FIT CLUSTERING MODELS (MAIN + ROBUSTNESSES)
# ============================================================

# --- Main estimator ---
# k-medoids is preferred because it is less sensitive to outliers
if HAS_KMEDOIDS:
    model_main, labels_main = fit_kmedoids(Z, K_STAR, RANDOM_STATE)
    main_name = "kmedoids"
else:
    model_main, labels_main = fit_kmeans(Z, K_STAR, RANDOM_STATE)
    main_name = "kmeans_fallback"

# --- Robustness estimator 1: k-means ---
_, labels_km = fit_kmeans(Z, K_STAR, RANDOM_STATE)

# --- Robustness estimator 2: HDBSCAN (if installed) ---
labels_hdb = None
if HAS_HDBSCAN:
    _, labels_hdb = fit_hdbscan(Z)

# Build an output table of cluster labels for mapping / merging
df_labels = df[
    ["year", "code_commune", "commune_name", "province_prefecture", "region", "n_households"]
].copy()
df_labels["cluster_main"] = labels_main
df_labels["cluster_kmeans"] = labels_km
if labels_hdb is not None:
    df_labels["cluster_hdbscan"] = labels_hdb

# Merge labels back to the full readable dataset
df_with_clusters = df.merge(
    df_labels[["code_commune", "cluster_main", "cluster_kmeans"] + (["cluster_hdbscan"] if labels_hdb is not None else [])],
    on="code_commune",
    how="left",
)

# Export mapping-ready dataset (join to shapefile in QGIS on code_commune)
df_with_clusters.to_csv(os.path.join(OUT_DIR, "communes_rgph_2024_with_clusters.csv"), index=False)


# ============================================================
# 11) PAPER TABLES (DESCRIPTIVES, PROFILES, METHOD CONCORDANCE)
# ============================================================

# --- T2: 2024 descriptives for readable variables (mean + IQR) ---
rows = []
for c in READABLE_VARS:
    x = pd.to_numeric(df_with_clusters[c], errors="coerce")
    rows.append(
        {
            "year": YEAR_KEEP,
            "variable": c,
            "mean": float(np.nanmean(x)),
            "p25": float(np.nanpercentile(x.dropna(), 25)) if x.notna().any() else np.nan,
            "median": float(np.nanmedian(x)),
            "p75": float(np.nanpercentile(x.dropna(), 75)) if x.notna().any() else np.nan,
            "missing_share": float(x.isna().mean()),
        }
    )
T2 = pd.DataFrame(rows)
T2.to_csv(os.path.join(OUT_DIR, "T2_descriptives_2024.csv"), index=False)

# --- T3: Cluster profiles in original units (optionally household-weighted) ---
T3 = weighted_cluster_profiles(
    df_with_clusters,
    label_col="cluster_main",
    vars_readable=READABLE_VARS,
    weight_col=WEIGHT_COL if USE_WEIGHTS_IN_PROFILES else None,
)
T3.to_csv(os.path.join(OUT_DIR, "T3_cluster_profiles_2024.csv"), index=False)

# --- T5: Agreement between methods (main vs k-means; optionally main vs HDBSCAN) ---
rows = [
    {
        "compare": "main_vs_kmeans",
        "ARI": adjusted_rand_score(labels_main, labels_km),
        "NMI": normalized_mutual_info_score(labels_main, labels_km),
    }
]
if labels_hdb is not None:
    rows.append(
        {
            "compare": "main_vs_hdbscan",
            "ARI": adjusted_rand_score(labels_main, labels_hdb),
            "NMI": normalized_mutual_info_score(labels_main, labels_hdb),
        }
    )
pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "T5_concordance_methods_2024.csv"), index=False)


# ============================================================
# 12) STABILITY VIA SUBSAMPLING (BOOTSTRAP-LIKE)
# ============================================================

# We refit the main estimator on subsamples and compare to the full-sample labels.
stab_main = bootstrap_stability(
    Z,
    labels_ref=labels_main,
    fit_fun=fit_kmedoids if HAS_KMEDOIDS else fit_kmeans,
    k=K_STAR,
    n_boot=N_BOOT,
    frac=SUBSAMPLE_FRAC,
    random_state=RANDOM_STATE,
)
stab_main.to_csv(os.path.join(OUT_DIR, "T5_bootstrap_stability_main_2024.csv"), index=False)

# Plot ARI distribution over subsamples (paper-ready stability figure)
plt.figure()
plt.plot(stab_main["boot"], stab_main["ARI"], marker="o", linestyle="None")
plt.axhline(stab_main["ARI"].mean(), linestyle="--")
plt.xlabel("Subsample iteration")
plt.ylabel("ARI (agreement)")
plt.title("Clustering stability (ARI) — 2024")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "F_stability_ARI_bootstrap_2024.png"), dpi=200)
plt.close()


# ============================================================
# 13) EXAMPLE FIGURES FOR INTERPRETATION (COOKING FUEL)
# ============================================================

# Create interpretable aggregate indicators for narrative:
# biomass = wood + charcoal; modern = gas + electricity
df_with_clusters["biomass_pct"] = df_with_clusters["cook_wood_pct"] + df_with_clusters["cook_charcoal_pct"]
df_with_clusters["modern_pct"] = df_with_clusters["cook_gas_pct"] + df_with_clusters["cook_elec_pct"]

# Boxplots by cluster (simple, interpretable paper figures)
plt.figure()
df_with_clusters.boxplot(column="biomass_pct", by="cluster_main")
plt.suptitle("")  # remove auto title from pandas
plt.title("Biomass cooking share (wood+charcoal) by cluster — 2024")
plt.xlabel("Cluster")
plt.ylabel("%")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "F_box_biomass_by_cluster_2024.png"), dpi=200)
plt.close()

plt.figure()
df_with_clusters.boxplot(column="cook_gas_pct", by="cluster_main")
plt.suptitle("")
plt.title("Gas cooking share by cluster — 2024")
plt.xlabel("Cluster")
plt.ylabel("%")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "F_box_gas_by_cluster_2024.png"), dpi=200)
plt.close()


# ============================================================
# 14) RGPH 2024 DESIGN ROBUSTNESS: STRATIFICATION BY COMMUNE SIZE
# ============================================================

# Motivation:
#   If long questionnaire is sampled at 20% in >=2000 household communes,
#   some shares may be noisier; stratified clustering checks whether the
#   structure is consistent across strata.

df_2024 = df_with_clusters.copy()
df_2024["size_stratum"] = np.where(
    df_2024["n_households"].astype(float) < DESIGN_THRESHOLD,
    "<2000",
    ">=2000",
)


def refit_on_subset(df_sub):
    """
    Refit the full pipeline (transforms + scaling + PCA + clustering) on a subset.
    This is used for stratified clustering robustness checks.
    """
    dsub = df_sub.copy()

    dsub["n_households_log"] = np.log(np.clip(dsub["n_households"].astype(float), 1, None))
    dsub["distance_log1p"] = np.log1p(pd.to_numeric(dsub[DIST_COL], errors="coerce"))

    for c in COMFORT_RATE_COLS:
        dsub[f"{c}__logit"] = logit_transform_pct(pd.to_numeric(dsub[c], errors="coerce"), eps=EPS)

    clr_parts_sub = []
    for _, cols in BLOCKS_COMPOSITIONAL.items():
        clr_parts_sub.append(clr_transform_block(dsub, cols, eps=EPS))
    X_clr_sub = pd.concat(clr_parts_sub, axis=1)

    X_sub = pd.concat(
        [
            dsub[["hh_size_mean", "persons_per_room_mean", "n_households_log", "distance_log1p"]],
            dsub[[f"{c}__logit" for c in COMFORT_RATE_COLS]],
            X_clr_sub,
        ],
        axis=1,
    )

    for c in X_sub.columns:
        X_sub[c] = X_sub[c].fillna(X_sub[c].median())

    Xs_sub, _ = robust_scale(X_sub)
    Z_sub, _, _, _, _ = pca_reduce(Xs_sub, variance_target=PCA_VARIANCE_TARGET, random_state=RANDOM_STATE)

    if HAS_KMEDOIDS:
        _, lab = fit_kmedoids(Z_sub, K_STAR, RANDOM_STATE)
    else:
        _, lab = fit_kmeans(Z_sub, K_STAR, RANDOM_STATE)

    return lab


# Run stratified re-fit; we just export summary info (you can extend to compare profiles)
design_rows = []
for stratum, g in df_2024.groupby("size_stratum"):
    _ = refit_on_subset(g)  # labels not merged here; purpose is feasibility check + extension point
    design_rows.append(
        {
            "stratum": stratum,
            "n": int(len(g)),
            "K_used": int(K_STAR),
            "method": "kmedoids" if HAS_KMEDOIDS else "kmeans",
            "note": "Refit pipeline within stratum; compare profiles as robustness.",
        }
    )
pd.DataFrame(design_rows).to_csv(os.path.join(OUT_DIR, "design2024_stratification_info.csv"), index=False)


# ============================================================
# 15) FINAL LOG (what to look at first)
# ============================================================

print("[DONE] 2024-only pipeline completed. Outputs in:", OUT_DIR)
print("Start here:")
print("  1) QA_block_sums_2024.csv (check compositional sums + renormalization necessity)")
print("  2) K_selection_metrics_2024.csv + figures (pick K and justify choice)")
print("  3) T3_cluster_profiles_2024.csv (interpret clusters with readable variables)")
print("  4) T5_bootstrap_stability_main_2024.csv (stability evidence)")
print("Mapping file:")
print("  - communes_rgph_2024_with_clusters.csv (join on code_commune in QGIS)")
