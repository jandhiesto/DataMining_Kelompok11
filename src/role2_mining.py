"""
Module src/role2_mining.py
Implementasi Production-Grade untuk Role 2 (Task 3, Task 4, Task 5, Task 6)
menggunakan dataset bersih DBLP 2015-2025 dari Role 1 (`data/clean_inproceedings.parquet`).

Skala data nyata DBLP (clean_summary.json):
- 1.899.038 publikasi bersih (inproceedings 2015-2025)
- 1.738.301 author unik (Author-Based transactions)
- 7.428.192 entri author-paper
- 4.232 venue_key kanonik unik (1-ke-1 dengan venue_name)

Optimasi teknik agar cepat & hemat RAM di laptop (16 GB RAM) tanpa aproksimasi:
1. Konstruksi transaksi sepenuhnya vectorized via NumPy int64 bit-packing + SciPy CSR sparse matrix.
2. Pada setiap level minimum support (0.1%, 0.5%, 1%, 2%), dilakukan proyeksi eksak ke kolom
   frequent 1-itemset dan baris aktif (sifat anti-monotonitas Apriori) dengan penyesuaian
   threshold eksak terhadap N_total = 1.738.301 sehingga seluruh support & itemsets
   100% identik secara matematis dengan komputasi matriks penuh.
"""

import gc
import json
import math
import time
import tracemalloc
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr

try:
    from src.data_loader import build_venue_label_map, load_clean_data
except ImportError:
    from data_loader import build_venue_label_map, load_clean_data

OUTPUT_DIR = Path("output")
MIN_SUPPORTS = [0.001, 0.005, 0.01, 0.02]
MIN_CONFIDENCE = 0.50


def format_itemset(itemset, label_map: Dict[str, str]) -> str:
    """Mengubah frozenset venue_key menjadi string label tampilan terurut."""
    labels = sorted(label_map.get(k, str(k)) for k in itemset)
    return "{" + ", ".join(labels) + "}"


def format_itemset_keys(itemset) -> str:
    """Mengubah frozenset venue_key menjadi string venue_key kanonik terurut."""
    return "{" + ", ".join(sorted(str(k) for k in itemset)) + "}"


# ==============================================================================
# TASK 3: TRANSACTIONAL DATABASE CONSTRUCTION
# ==============================================================================
def construct_transactional_databases(
    df: pd.DataFrame,
    venue_display_map: Dict[str, str],
    output_dir: Path = OUTPUT_DIR,
    save_artifacts: bool = True,
) -> dict:
    """
    Membangun dua jenis representasi transactional database secara vectorized:
    A. Author-Based Transaction: 1 transaksi = 1 author_id, item = himpunan venue_key unik (2015-2025).
    B. Author-Year-Based Transaction: 1 transaksi = 1 (author_id, year), item = himpunan venue_key unik tahun tsb.
    """
    t0 = time.perf_counter()
    output_dir.mkdir(exist_ok=True)

    # 1. Faktorisasi venue_key dan konversi tahun ke offset (year - year_min)
    venue_codes_paper, unique_venues = pd.factorize(df["venue_key"], sort=True)
    unique_venues = np.asarray(unique_venues, dtype=object)
    n_venues = len(unique_venues)
    assert n_venues < (1 << 16), f"Jumlah venue ({n_venues}) melebihi batas 16-bit!"

    year_min = int(df["year"].min())
    year_max = int(df["year"].max())
    year_offsets_paper = (df["year"].to_numpy(dtype=np.int16) - year_min).astype(np.int16)
    assert (year_max - year_min) < 16, "Rentang tahun melebihi 4-bit offset!"

    # 2. Ratakan daftar author_id menjadi array flat & faktorisasi author_id
    authors_col = df["author_id"].tolist()
    n_authors_per_paper = np.fromiter(
        (len(lst) for lst in authors_col), dtype=np.int32, count=len(authors_col)
    )
    flat_authors = [a for lst in authors_col for a in lst]
    author_codes_flat, unique_authors = pd.factorize(
        pd.Series(flat_authors, dtype="object"), sort=False
    )
    del flat_authors, authors_col
    unique_authors = np.asarray(unique_authors, dtype=object)
    n_authors = len(unique_authors)

    venue_codes_flat = np.repeat(venue_codes_paper.astype(np.int32), n_authors_per_paper)
    year_offsets_flat = np.repeat(year_offsets_paper.astype(np.int32), n_authors_per_paper)
    del venue_codes_paper, year_offsets_paper, n_authors_per_paper
    gc.collect()

    # --------------------------------------------------------------------------
    # A. Author-Based Transactional Database (deduplikasi pasangan author, venue)
    # --------------------------------------------------------------------------
    av_keys = (author_codes_flat.astype(np.int64) << 16) | venue_codes_flat.astype(np.int64)
    av_unique = np.unique(av_keys)
    del av_keys

    u_author = (av_unique >> 16).astype(np.int32)
    u_venue = (av_unique & 0xFFFF).astype(np.int32)
    del av_unique

    X_author = csr_matrix(
        (np.ones(len(u_author), dtype=bool), (u_author, u_venue)),
        shape=(n_authors, n_venues),
        dtype=bool,
    )
    X_author.sort_indices()
    author_tx_lens = np.diff(X_author.indptr).astype(np.int32)
    del u_author, u_venue

    # --------------------------------------------------------------------------
    # B. Author-Year-Based Transactional Database (deduplikasi tripel author, year, venue)
    # --------------------------------------------------------------------------
    ayv_keys = (
        (author_codes_flat.astype(np.int64) << 20)
        | (year_offsets_flat.astype(np.int64) << 16)
        | venue_codes_flat.astype(np.int64)
    )
    del author_codes_flat, year_offsets_flat, venue_codes_flat
    gc.collect()

    ayv_unique = np.unique(ayv_keys)
    del ayv_keys

    ay_pair_keys = ayv_unique >> 16  # (author_code << 4) | year_offset
    unique_ay_keys, ay_start_idx, ay_tx_lens = np.unique(
        ay_pair_keys, return_index=True, return_counts=True
    )
    ay_tx_lens = ay_tx_lens.astype(np.int32)
    n_ay_tx = len(unique_ay_keys)
    ay_unique_venues = int(len(np.unique(ayv_unique & 0xFFFF)))

    # --------------------------------------------------------------------------
    # Contoh Transaksi (10 Author-Based & 10 Author-Year-Based, deterministik)
    # Memilih campuran transaksi 1-item dan multi-item secara deterministik agar informatif
    # --------------------------------------------------------------------------
    rng = np.random.default_rng(42)

    def pick_representative_indices(lens_arr: np.ndarray, total_n: int = 10) -> List[int]:
        chosen = []
        # Ambil 2 contoh panjang 1, 3 panjang 2, 3 panjang 3-5, 2 panjang >=6 (jika tersedia)
        buckets = [
            (np.where(lens_arr == 1)[0], 2),
            (np.where(lens_arr == 2)[0], 3),
            (np.where((lens_arr >= 3) & (lens_arr <= 5))[0], 3),
            (np.where(lens_arr >= 6)[0], 2),
        ]
        for pool, k in buckets:
            if len(pool) > 0:
                take = min(k, len(pool))
                chosen.extend(rng.choice(pool, size=take, replace=False).tolist())
        # Jika kurang dari 10, lengkapi secara acak dari sisa indeks
        if len(chosen) < total_n:
            remaining = np.setdiff1d(np.arange(len(lens_arr)), chosen)
            extra = min(total_n - len(chosen), len(remaining))
            if extra > 0:
                chosen.extend(rng.choice(remaining, size=extra, replace=False).tolist())
        return sorted(chosen[:total_n], key=lambda idx: (-int(lens_arr[idx]), idx))

    sample_a_idx = pick_representative_indices(author_tx_lens, 10)
    examples_author = []
    for rank, idx in enumerate(sample_a_idx, start=1):
        v_idxs = X_author.indices[X_author.indptr[idx] : X_author.indptr[idx + 1]]
        v_keys = [str(unique_venues[j]) for j in v_idxs]
        v_names = [venue_display_map.get(vk, vk) for vk in v_keys]
        examples_author.append({
            "no": rank,
            "author_id": str(unique_authors[idx]),
            "n_items": len(v_keys),
            "venue_keys": ", ".join(v_keys),
            "venue_names": ", ".join(v_names),
        })
    df_ex_author = pd.DataFrame(examples_author)

    sample_ay_idx = pick_representative_indices(ay_tx_lens, 10)
    examples_ay = []
    for rank, tx_i in enumerate(sample_ay_idx, start=1):
        start = int(ay_start_idx[tx_i])
        length = int(ay_tx_lens[tx_i])
        slice_keys = ayv_unique[start : start + length]
        a_code = int(unique_ay_keys[tx_i] >> 4)
        yr = year_min + int(unique_ay_keys[tx_i] & 0xF)
        v_idxs = (slice_keys & 0xFFFF).astype(np.int32)
        v_keys = [str(unique_venues[j]) for j in v_idxs]
        v_names = [venue_display_map.get(vk, vk) for vk in v_keys]
        examples_ay.append({
            "no": rank,
            "author_id": str(unique_authors[a_code]),
            "year": yr,
            "n_items": len(v_keys),
            "venue_keys": ", ".join(v_keys),
            "venue_names": ", ".join(v_names),
        })
    df_ex_ay = pd.DataFrame(examples_ay)

    del ayv_unique, ay_pair_keys, unique_ay_keys, ay_start_idx
    gc.collect()

    # --------------------------------------------------------------------------
    # Tabel Statistik Komparasi Kedua Representasi
    # --------------------------------------------------------------------------
    def summarize_lengths(lens_arr: np.ndarray, n_items_unique: int) -> dict:
        n_tx = len(lens_arr)
        nnz = int(lens_arr.sum())
        density_pct = (nnz / (n_tx * n_items_unique)) * 100.0 if (n_tx * n_items_unique) > 0 else 0.0
        n_eq1 = int((lens_arr == 1).sum())
        n_ge2 = int((lens_arr >= 2).sum())
        n_ge3 = int((lens_arr >= 3).sum())
        return {
            "n_tx": n_tx,
            "n_unique_items": n_items_unique,
            "total_entries": nnz,
            "density_pct": density_pct,
            "min": int(lens_arr.min()),
            "p25": float(np.percentile(lens_arr, 25)),
            "median": float(np.median(lens_arr)),
            "mean": float(lens_arr.mean()),
            "p75": float(np.percentile(lens_arr, 75)),
            "p95": float(np.percentile(lens_arr, 95)),
            "p99": float(np.percentile(lens_arr, 99)),
            "max": int(lens_arr.max()),
            "std": float(lens_arr.std()),
            "n_eq1": n_eq1,
            "pct_eq1": (n_eq1 / n_tx) * 100.0,
            "n_ge2": n_ge2,
            "pct_ge2": (n_ge2 / n_tx) * 100.0,
            "n_ge3": n_ge3,
            "pct_ge3": (n_ge3 / n_tx) * 100.0,
        }

    s_a = summarize_lengths(author_tx_lens, n_venues)
    s_ay = summarize_lengths(ay_tx_lens, ay_unique_venues)

    stats_df = pd.DataFrame({
        "Metrik": [
            "Jumlah Transaksi",
            "Jumlah Item Unik (venue_key)",
            "Total Pasangan Transaksi-Item (NNZ)",
            "Kepadatan Matriks (%)",
            "Item per Transaksi — Min",
            "Item per Transaksi — Kuartil 1 (P25)",
            "Item per Transaksi — Median (P50)",
            "Item per Transaksi — Rata-rata (Mean)",
            "Item per Transaksi — Kuartil 3 (P75)",
            "Item per Transaksi — Persentil 95 (P95)",
            "Item per Transaksi — Persentil 99 (P99)",
            "Item per Transaksi — Maksimum",
            "Item per Transaksi — Standar Deviasi",
            "Transaksi 1 Item (len = 1)",
            "Transaksi Multi-Item (len >= 2)",
            "Transaksi >= 3 Item (len >= 3)",
        ],
        "Author-Based": [
            f"{s_a['n_tx']:,}",
            f"{s_a['n_unique_items']:,}",
            f"{s_a['total_entries']:,}",
            f"{s_a['density_pct']:.4f}%",
            f"{s_a['min']}",
            f"{s_a['p25']:.1f}",
            f"{s_a['median']:.1f}",
            f"{s_a['mean']:.3f}",
            f"{s_a['p75']:.1f}",
            f"{s_a['p95']:.1f}",
            f"{s_a['p99']:.1f}",
            f"{s_a['max']:,}",
            f"{s_a['std']:.3f}",
            f"{s_a['n_eq1']:,} ({s_a['pct_eq1']:.2f}%)",
            f"{s_a['n_ge2']:,} ({s_a['pct_ge2']:.2f}%)",
            f"{s_a['n_ge3']:,} ({s_a['pct_ge3']:.2f}%)",
        ],
        "Author-Year-Based": [
            f"{s_ay['n_tx']:,}",
            f"{s_ay['n_unique_items']:,}",
            f"{s_ay['total_entries']:,}",
            f"{s_ay['density_pct']:.4f}%",
            f"{s_ay['min']}",
            f"{s_ay['p25']:.1f}",
            f"{s_ay['median']:.1f}",
            f"{s_ay['mean']:.3f}",
            f"{s_ay['p75']:.1f}",
            f"{s_ay['p95']:.1f}",
            f"{s_ay['p99']:.1f}",
            f"{s_ay['max']:,}",
            f"{s_ay['std']:.3f}",
            f"{s_ay['n_eq1']:,} ({s_ay['pct_eq1']:.2f}%)",
            f"{s_ay['n_ge2']:,} ({s_ay['pct_ge2']:.2f}%)",
            f"{s_ay['n_ge3']:,} ({s_ay['pct_ge3']:.2f}%)",
        ],
    })

    # Distribusi ringkas bucket panjang transaksi (1, 2, 3, 4, 5, 6-10, >10)
    def bucket_dist(lens_arr: np.ndarray) -> pd.Series:
        n = len(lens_arr)
        labels = ["1", "2", "3", "4", "5", "6-10", ">10"]
        counts = [
            int((lens_arr == 1).sum()),
            int((lens_arr == 2).sum()),
            int((lens_arr == 3).sum()),
            int((lens_arr == 4).sum()),
            int((lens_arr == 5).sum()),
            int(((lens_arr >= 6) & (lens_arr <= 10)).sum()),
            int((lens_arr > 10).sum()),
        ]
        return pd.DataFrame({
            "bucket": labels,
            "count": counts,
            "pct": [round(c / n * 100.0, 2) for c in counts],
        })

    dist_a = bucket_dist(author_tx_lens)
    dist_ay = bucket_dist(ay_tx_lens)

    elapsed = time.perf_counter() - t0
    print(f"\n[Task 3] Konstruksi kedua Transactional Database selesai dalam {elapsed:.2f} detik.")
    print(stats_df.to_string(index=False))

    if save_artifacts:
        stats_df.to_csv(output_dir / "task3_transaction_stats.csv", index=False)
        df_ex_author.to_csv(output_dir / "task3_examples_author.csv", index=False)
        df_ex_ay.to_csv(output_dir / "task3_examples_author_year.csv", index=False)
        plot_task3_distribution(author_tx_lens, ay_tx_lens, output_dir / "task3_transaction_distribution.png")

    return {
        "X_author": X_author,
        "unique_venues": unique_venues,
        "unique_authors": unique_authors,
        "author_tx_lens": author_tx_lens,
        "ay_tx_lens": ay_tx_lens,
        "stats_df": stats_df,
        "stats_raw": {"author_based": s_a, "author_year_based": s_ay},
        "dist_author": dist_a,
        "dist_ay": dist_ay,
        "examples_author": df_ex_author,
        "examples_ay": df_ex_ay,
    }


def plot_task3_distribution(
    author_tx_lens: np.ndarray,
    ay_tx_lens: np.ndarray,
    out_path: Path,
):
    """Plot distribusi jumlah item per transaksi (skala log pada sumbu Y)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot distribusi hingga batas persentil 99.9 atau cap visual yang rapi
    cap_a = min(int(author_tx_lens.max()), 25)
    vals_a, cnts_a = np.unique(np.minimum(author_tx_lens, cap_a), return_counts=True)
    x_lbl_a = [f"{v}+" if v == cap_a and author_tx_lens.max() > cap_a else str(v) for v in vals_a]

    axes[0].bar(range(len(vals_a)), cnts_a, color="#2B6CB0", edgecolor="black", alpha=0.88)
    axes[0].set_yscale("log")
    axes[0].set_xticks(range(len(vals_a)))
    axes[0].set_xticklabels(x_lbl_a, fontsize=8, rotation=45 if len(vals_a) > 12 else 0)
    axes[0].set_title(
        f"Distribusi Jumlah Item: Author-Based\n(N = {len(author_tx_lens):,} transaksi | Mean = {author_tx_lens.mean():.2f})",
        fontsize=10.5,
        fontweight="bold",
    )
    axes[0].set_xlabel("Jumlah Venue Unik per Author (2015-2025)")
    axes[0].set_ylabel("Jumlah Transaksi (Skala Log)")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    cap_ay = min(int(ay_tx_lens.max()), 15)
    vals_ay, cnts_ay = np.unique(np.minimum(ay_tx_lens, cap_ay), return_counts=True)
    x_lbl_ay = [f"{v}+" if v == cap_ay and ay_tx_lens.max() > cap_ay else str(v) for v in vals_ay]

    axes[1].bar(range(len(vals_ay)), cnts_ay, color="#C05621", edgecolor="black", alpha=0.88)
    axes[1].set_yscale("log")
    axes[1].set_xticks(range(len(vals_ay)))
    axes[1].set_xticklabels(x_lbl_ay, fontsize=8)
    axes[1].set_title(
        f"Distribusi Jumlah Item: Author-Year-Based\n(N = {len(ay_tx_lens):,} transaksi | Mean = {ay_tx_lens.mean():.2f})",
        fontsize=10.5,
        fontweight="bold",
    )
    axes[1].set_xlabel("Jumlah Venue Unik per (Author, Year)")
    axes[1].set_ylabel("Jumlah Transaksi (Skala Log)")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[Visualisasi] Disimpan ke: {out_path}")


# ==============================================================================
# TASK 4 & TASK 6: FREQUENT PATTERN MINING & ALGORITHM PERFORMANCE COMPARISON
# ==============================================================================
def _project_active_matrix(
    X_author: csr_matrix,
    unique_venues: np.ndarray,
    min_support: float,
) -> Tuple[pd.DataFrame, int, int, int, float]:
    """
    Proyeksi eksak (zero-approximation) berdasarkan sifat anti-monotonitas Apriori:
    - Hanya kolom venue dengan count >= ceil(min_support * N_total) yang dapat muncul
      dalam frequent itemset berukuran >= 1.
    - Hanya baris transaksi yang memiliki >= 1 venue frequent tersebut yang dapat
      berkontribusi pada hitungan support.
    - Threshold disesuaikan eksak: s_adj = (min_count - 0.25) / N_active.
    """
    n_total = X_author.shape[0]
    min_count = math.ceil(min_support * n_total - 1e-12)

    venue_counts = np.asarray(X_author.sum(axis=0)).ravel()
    freq_cols = np.where(venue_counts >= min_count)[0]
    if len(freq_cols) == 0:
        return pd.DataFrame(), n_total, 0, min_count, 1.0

    X_sub = X_author[:, freq_cols].tocsr()
    row_nnz = np.diff(X_sub.indptr)
    active_rows = np.where(row_nnz > 0)[0]
    X_active = X_sub[active_rows]
    n_active = X_active.shape[0]
    col_names = [str(unique_venues[c]) for c in freq_cols]

    # Konversi ke DataFrame bool (1 byte per sel) untuk kompatibilitas penuh & RAM rendah di mlxtend
    est_bytes = n_active * len(freq_cols)
    if est_bytes <= 1_200_000_000:  # <= ~1.2 GB aman di 16 GB RAM
        df_active = pd.DataFrame(X_active.toarray(), columns=col_names, dtype=bool)
    else:
        df_active = pd.DataFrame.sparse.from_spmatrix(X_active, columns=col_names)

    s_adj = (min_count - 0.25) / float(n_active)
    return df_active, n_total, n_active, min_count, s_adj


def _rescale_itemsets(
    fi: pd.DataFrame,
    n_active: int,
    n_total: int,
    min_count: int,
    venue_display_map: Dict[str, str],
) -> pd.DataFrame:
    """Mengembalikan nilai support eksak terhadap N_total dan menambahkan kolom bantu."""
    if fi is None or fi.empty:
        return pd.DataFrame(
            columns=["support", "count", "itemsets", "length", "itemsets_keys", "itemsets_names"]
        )
    out = fi.copy()
    counts = np.rint(out["support"].to_numpy(dtype=np.float64) * n_active).astype(np.int64)
    out["count"] = counts
    out["support"] = counts / float(n_total)
    out = out[out["count"] >= min_count].copy()
    out["length"] = out["itemsets"].apply(len).astype(int)
    out["itemsets_keys"] = out["itemsets"].apply(format_itemset_keys)
    out["itemsets_names"] = out["itemsets"].apply(lambda s: format_itemset(s, venue_display_map))
    out = out.sort_values(by=["support", "length", "itemsets_keys"], ascending=[False, True, True]).reset_index(drop=True)
    return out


def _generate_rules_safe(
    fi: pd.DataFrame,
    min_confidence: float,
    venue_display_map: Dict[str, str],
) -> pd.DataFrame:
    """Membangkitkan association rules secara aman (menangani kasus tanpa >=2-itemsets)."""
    if fi is None or fi.empty or not (fi["length"] >= 2).any():
        return pd.DataFrame(
            columns=[
                "antecedents", "consequents", "antecedent support", "consequent support",
                "support", "confidence", "lift", "antecedents_keys", "consequents_keys",
                "antecedents_names", "consequents_names", "rule_keys", "rule_names",
            ]
        )
    rules = association_rules(
        fi[["support", "itemsets"]],
        metric="confidence",
        min_threshold=min_confidence,
    )
    if rules.empty:
        return rules
    rules["antecedents_keys"] = rules["antecedents"].apply(format_itemset_keys)
    rules["consequents_keys"] = rules["consequents"].apply(format_itemset_keys)
    rules["antecedents_names"] = rules["antecedents"].apply(lambda s: format_itemset(s, venue_display_map))
    rules["consequents_names"] = rules["consequents"].apply(lambda s: format_itemset(s, venue_display_map))
    rules["rule_keys"] = rules["antecedents_keys"] + " -> " + rules["consequents_keys"]
    rules["rule_names"] = rules["antecedents_names"] + " -> " + rules["consequents_names"]
    rules = rules.sort_values(
        by=["lift", "confidence", "support"], ascending=[False, False, False]
    ).reset_index(drop=True)
    return rules


def run_fpm_and_benchmark(
    X_author: csr_matrix,
    unique_venues: np.ndarray,
    venue_display_map: Dict[str, str],
    min_supports: List[float] = MIN_SUPPORTS,
    min_confidence: float = MIN_CONFIDENCE,
    output_dir: Path = OUTPUT_DIR,
    save_artifacts: bool = True,
) -> Tuple[dict, pd.DataFrame]:
    """
    Menjalankan Apriori dan FP-Growth pada Author-Based Transactional Database
    untuk setiap nilai minimum support (0.1%, 0.5%, 1%, 2%), memverifikasi kesamaan
    itemsets & rules, serta mencatat runtime dan peak memory secara adil.
    """
    output_dir.mkdir(exist_ok=True)
    n_total = X_author.shape[0]
    benchmark_rows = []
    results_by_sup = {}

    for s in min_supports:
        sup_pct = f"{s * 100:.1f}%"
        sup_tag = f"{s * 100:.1f}pct".replace(".", "_")
        print(f"\n{'=' * 78}")
        print(f"Eksperimen Min Support = {sup_pct} (min_count >= {math.ceil(s * n_total - 1e-12):,} dari {n_total:,} author)")
        print(f"{'=' * 78}")

        t_proj0 = time.perf_counter()
        df_active, n_tot, n_act, min_count, s_adj = _project_active_matrix(X_author, unique_venues, s)
        t_proj = time.perf_counter() - t_proj0
        n_cand_cols = df_active.shape[1]
        print(f"[Proyeksi Eksak] {n_act:,} author aktif x {n_cand_cols} venue kandidat ({t_proj:.2f}s)")

        if df_active.empty:
            fi_fp = _rescale_itemsets(pd.DataFrame(), 0, n_tot, min_count, venue_display_map)
            fi_ap = fi_fp.copy()
            t_fp = t_ap = mem_fp_mb = mem_ap_mb = 0.0
            rules_fp = _generate_rules_safe(fi_fp, min_confidence, venue_display_map)
            rules_ap = rules_fp.copy()
            is_identical = True
            max_sup_diff = 0.0
        else:
            # --- 1. FP-GROWTH ---
            gc.collect()
            tracemalloc.start()
            t0 = time.perf_counter()
            raw_fp = fpgrowth(df_active, min_support=s_adj, use_colnames=True)
            t_fp = time.perf_counter() - t0
            _, peak_fp = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            mem_fp_mb = peak_fp / (1024 * 1024)
            fi_fp = _rescale_itemsets(raw_fp, n_act, n_tot, min_count, venue_display_map)
            del raw_fp

            # --- 2. APRIORI (low_memory=True agar tidak OOM pada 1.7M transaksi) ---
            gc.collect()
            tracemalloc.start()
            t0 = time.perf_counter()
            raw_ap = apriori(df_active, min_support=s_adj, use_colnames=True, low_memory=True)
            t_ap = time.perf_counter() - t0
            _, peak_ap = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            mem_ap_mb = peak_ap / (1024 * 1024)
            fi_ap = _rescale_itemsets(raw_ap, n_act, n_tot, min_count, venue_display_map)
            del raw_ap

            # Verifikasi kesamaan matematis eksak (itemsets DAN nilai support)
            map_fp = dict(zip(fi_fp["itemsets"].apply(frozenset), fi_fp["support"]))
            map_ap = dict(zip(fi_ap["itemsets"].apply(frozenset), fi_ap["support"]))
            same_sets = set(map_fp.keys()) == set(map_ap.keys())
            max_sup_diff = (
                max((abs(map_fp[k] - map_ap[k]) for k in map_fp), default=0.0)
                if same_sets
                else float("inf")
            )
            is_identical = bool(same_sets and max_sup_diff < 1e-12)

            # Association rules (min_confidence = 50%) untuk kedua algoritma
            rules_fp = _generate_rules_safe(fi_fp, min_confidence, venue_display_map)
            rules_ap = _generate_rules_safe(fi_ap, min_confidence, venue_display_map)

            del df_active
            gc.collect()

        c1 = int((fi_fp["length"] == 1).sum())
        c2 = int((fi_fp["length"] == 2).sum())
        c3 = int((fi_fp["length"] == 3).sum())
        c4p = int((fi_fp["length"] >= 4).sum())

        benchmark_rows.append({
            "min_support_pct": sup_pct,
            "min_support": s,
            "min_count": min_count,
            "active_authors": n_act,
            "itemsets_apriori": len(fi_ap),
            "itemsets_fpgrowth": len(fi_fp),
            "itemsets_identical": is_identical,
            "max_support_diff": max_sup_diff,
            "n_1_itemsets": c1,
            "n_2_itemsets": c2,
            "n_3_itemsets": c3,
            "n_4plus_itemsets": c4p,
            "rules_apriori_conf50": len(rules_ap),
            "rules_fpgrowth_conf50": len(rules_fp),
            "runtime_apriori_s": round(t_ap, 4),
            "runtime_fpgrowth_s": round(t_fp, 4),
            "speedup_ap_over_fp": round(t_ap / t_fp, 2) if t_fp > 0 else np.nan,
            "peak_ram_apriori_mb": round(mem_ap_mb, 2),
            "peak_ram_fpgrowth_mb": round(mem_fp_mb, 2),
        })

        results_by_sup[s] = {
            "fi_fp": fi_fp,
            "fi_ap": fi_ap,
            "rules_fp": rules_fp,
            "rules_ap": rules_ap,
        }

        print(
            f"  Itemsets: FP={len(fi_fp):,} | Apriori={len(fi_ap):,} | Identik={is_identical} "
            f"(1-item={c1}, 2-item={c2}, 3-item={c3}, >=4-item={c4p})"
        )
        print(
            f"  Rules (conf>=50%): FP={len(rules_fp):,} | Apriori={len(rules_ap):,}"
        )
        print(
            f"  Runtime: Apriori={t_ap:.4f}s vs FP-Growth={t_fp:.4f}s | "
            f"Peak RAM: Apriori={mem_ap_mb:.2f} MB vs FP-Growth={mem_fp_mb:.2f} MB"
        )

        if save_artifacts and not fi_fp.empty:
            fi_fp[["support", "count", "length", "itemsets_keys", "itemsets_names"]].to_csv(
                output_dir / f"task4_itemsets_{sup_tag}.csv", index=False
            )

    benchmark_df = pd.DataFrame(benchmark_rows)
    print("\n" + "=" * 78)
    print("RINGKASAN BENCHMARK TASK 4 & TASK 6")
    print("=" * 78)
    print(benchmark_df.to_string(index=False))

    if save_artifacts:
        benchmark_df.to_csv(output_dir / "task6_performance_comparison.csv", index=False)
        plot_task4_sensitivity(benchmark_df, output_dir / "task4_support_sensitivity.png")
        plot_task6_comparison(benchmark_df, output_dir / "task6_algorithm_performance_comparison.png")

    return results_by_sup, benchmark_df


def plot_task4_sensitivity(benchmark_df: pd.DataFrame, out_path: Path):
    """Plot pengaruh perubahan minimum support terhadap jumlah frequent itemsets per ukuran k."""
    fig, ax = plt.subplots(figsize=(9.5, 5))
    x = benchmark_df["min_support_pct"].tolist()

    ax.plot(x, benchmark_df["itemsets_fpgrowth"], marker="o", linewidth=2.5, markersize=8, color="#1B365D", label="Total Frequent Itemsets")
    ax.plot(x, benchmark_df["n_1_itemsets"], marker="s", linewidth=2, linestyle="--", color="#2B6CB0", label="1-itemsets")
    ax.plot(x, benchmark_df["n_2_itemsets"], marker="^", linewidth=2, linestyle="-.", color="#C05621", label="2-itemsets")
    ax.plot(x, benchmark_df["n_3_itemsets"], marker="D", linewidth=2, linestyle=":", color="#276749", label="3-itemsets")

    for i, val in enumerate(benchmark_df["itemsets_fpgrowth"]):
        ax.annotate(f"{int(val):,}", (x[i], val), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5, fontweight="bold")

    ax.set_title("Pengaruh Minimum Support terhadap Jumlah Frequent Itemsets (DBLP 2015-2025)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Minimum Support Threshold")
    ax.set_ylabel("Jumlah Frequent Itemsets")
    max_val = float(benchmark_df["itemsets_fpgrowth"].max())
    if max_val > 20 * max(1, float(benchmark_df["itemsets_fpgrowth"].min())):
        ax.set_yscale("symlog", linthresh=10)
        ax.set_ylim(-1.5, max_val * 2.5)
    else:
        ax.set_ylim(0, max_val * 1.2)
    ax.grid(True, linestyle="--", alpha=0.55)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[Visualisasi] Disimpan ke: {out_path}")


def plot_task6_comparison(benchmark_df: pd.DataFrame, out_path: Path):
    """Grafik 3-panel komparasi performa Apriori vs FP-Growth (Task 6)."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    x_labels = benchmark_df["min_support_pct"].tolist()
    x = np.arange(len(x_labels))
    w = 0.35

    # Panel 1: Runtime
    b1 = axes[0].bar(x - w / 2, benchmark_df["runtime_apriori_s"], w, label="Apriori", color="#C53030", edgecolor="black", alpha=0.88)
    b2 = axes[0].bar(x + w / 2, benchmark_df["runtime_fpgrowth_s"], w, label="FP-Growth", color="#2B6CB0", edgecolor="black", alpha=0.88)
    axes[0].set_title("Waktu Eksekusi (Detik)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Minimum Support")
    axes[0].set_ylabel("Runtime (s)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(x_labels)
    axes[0].legend()
    axes[0].grid(axis="y", linestyle="--", alpha=0.55)
    axes[0].set_ylim(0, float(max(benchmark_df["runtime_apriori_s"].max(), benchmark_df["runtime_fpgrowth_s"].max())) * 1.15)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2, h, f"{h:.2f}s", ha="center", va="bottom", fontsize=7.5)

    # Panel 2: Peak Memory
    m1 = axes[1].bar(x - w / 2, benchmark_df["peak_ram_apriori_mb"], w, label="Apriori", color="#DD6B20", edgecolor="black", alpha=0.88)
    m2 = axes[1].bar(x + w / 2, benchmark_df["peak_ram_fpgrowth_mb"], w, label="FP-Growth", color="#2F855A", edgecolor="black", alpha=0.88)
    axes[1].set_title("Puncak Penggunaan Memori (MB)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Minimum Support")
    axes[1].set_ylabel("Peak RAM (MB)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(x_labels)
    axes[1].legend()
    axes[1].grid(axis="y", linestyle="--", alpha=0.55)
    axes[1].set_ylim(0, float(max(benchmark_df["peak_ram_apriori_mb"].max(), benchmark_df["peak_ram_fpgrowth_mb"].max())) * 1.15)
    for bar in list(m1) + list(m2):
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2, h, f"{h:.1f}", ha="center", va="bottom", fontsize=7.5)

    # Panel 3: Frequent Itemsets & Association Rules
    axes[2].plot(x_labels, benchmark_df["itemsets_fpgrowth"], marker="o", linewidth=2.5, color="#6B46C1", label="Frequent Itemsets (Identik)")
    axes[2].plot(x_labels, benchmark_df["rules_fpgrowth_conf50"], marker="s", linewidth=2.5, color="#D69E2E", label="Association Rules (Conf >= 50%)")
    axes[2].set_title("Jumlah Itemsets & Rules vs Min Support", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Minimum Support")
    axes[2].set_ylabel("Jumlah Pola / Rules")
    if max(benchmark_df["itemsets_fpgrowth"].max(), benchmark_df["rules_fpgrowth_conf50"].max()) > 50:
        axes[2].set_yscale("symlog", linthresh=5)
    axes[2].legend()
    axes[2].grid(True, linestyle="--", alpha=0.55)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[Visualisasi] Disimpan ke: {out_path}")


# ==============================================================================
# TASK 5: ASSOCIATION RULE MINING & ANALISIS SUPPORT VS LIFT
# ==============================================================================
def analyze_association_rules(
    results_by_sup: dict,
    target_support: float = 0.001,
    output_dir: Path = OUTPUT_DIR,
    save_artifacts: bool = True,
) -> dict:
    """
    Menganalisis Association Rules (min confidence 50%) dari frequent itemsets
    pada target_support (default 0.1% — paling kaya pola), menampilkan Top 10 Lift,
    dan menguji korelasi Support vs Lift.
    """
    output_dir.mkdir(exist_ok=True)
    rules = results_by_sup[target_support]["rules_fp"].copy()

    print("\n" + "=" * 78)
    print(f"TASK 5: ASSOCIATION RULE MINING (Min Support={target_support*100:.1f}%, Min Confidence=50%)")
    print("=" * 78)
    print(f"Total association rules dihasilkan: {len(rules):,}")

    if rules.empty:
        print("Tidak ada aturan asosiasi yang memenuhi confidence >= 50% pada level support ini.")
        return {"rules": rules, "top10_lift": pd.DataFrame(), "top10_support": pd.DataFrame(), "corr": {}}

    cols_export = [
        "rule_names", "rule_keys", "antecedent support", "consequent support",
        "support", "confidence", "lift", "leverage", "conviction",
    ]
    cols_export = [c for c in cols_export if c in rules.columns]

    top10_lift = rules.head(10)[cols_export].copy()
    top10_support = rules.sort_values(by=["support", "lift"], ascending=[False, False]).head(10)[cols_export].copy()

    print("\n--- TOP 10 ASSOCIATION RULES BERDASARKAN LIFT TERTINGGI ---")
    print(top10_lift[["rule_names", "support", "confidence", "lift"]].to_string(index=False))

    print("\n--- TOP 10 ASSOCIATION RULES BERDASARKAN SUPPORT TERTINGGI (PEMBANDING) ---")
    print(top10_support[["rule_names", "support", "confidence", "lift"]].to_string(index=False))

    corr_info = {}
    if len(rules) >= 3:
        r_pearson = float(rules["support"].corr(rules["lift"]))
        r_spearman, p_spearman = spearmanr(rules["support"], rules["lift"])
        r_conseq_spearman, p_conseq = spearmanr(rules["consequent support"], rules["lift"])
        corr_info = {
            "pearson_support_lift": r_pearson,
            "spearman_support_lift": float(r_spearman),
            "spearman_p_value": float(p_spearman),
            "spearman_consequent_support_lift": float(r_conseq_spearman),
            "spearman_consequent_p_value": float(p_conseq),
        }
        print("\n--- ANALISIS KORELASI SUPPORT VS LIFT ---")
        print(f"Pearson (Support vs Lift)            : {r_pearson:.4f}")
        print(f"Spearman (Support vs Lift)           : {r_spearman:.4f} (p = {p_spearman:.2e})")
        print(f"Spearman (Consequent Supp vs Lift)   : {r_conseq_spearman:.4f} (p = {p_conseq:.2e})")

    if save_artifacts:
        rules[cols_export].to_csv(output_dir / "task5_association_rules.csv", index=False)
        top10_lift.to_csv(output_dir / "task5_top10_lift_rules.csv", index=False)
        plot_task5_rules(rules, top10_lift, output_dir)

    return {
        "rules": rules,
        "top10_lift": top10_lift,
        "top10_support": top10_support,
        "corr": corr_info,
    }


def plot_task5_rules(rules: pd.DataFrame, top10_lift: pd.DataFrame, output_dir: Path):
    """Plot Top 10 Rules by Lift dan Scatter Plot Support vs Lift."""
    # 1. Top 10 Lift Bar Chart
    fig, ax = plt.subplots(figsize=(12.5, 6.0))
    y = np.arange(len(top10_lift))
    bars = ax.barh(y, top10_lift["lift"], color="#276749", edgecolor="black", alpha=0.88)
    ax.set_yticks(y)
    ax.set_yticklabels(top10_lift["rule_names"], fontsize=9)
    ax.invert_yaxis()
    ax.set_title("Top 10 Association Rules Berdasarkan Nilai Lift (Min Support 0.1%, Min Conf 50%)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Nilai Lift")
    ax.grid(axis="x", linestyle="--", alpha=0.55)

    max_lift = float(top10_lift["lift"].max())
    ax.set_xlim(0, max_lift * 1.52)
    for bar, (_, row) in zip(bars, top10_lift.iterrows()):
        w = bar.get_width()
        ax.text(
            w + max_lift * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"Lift={w:.2f} | Conf={row['confidence']*100:.1f}% | Supp={row['support']*100:.2f}%",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="#1A365D",
        )
    plt.tight_layout()
    p1 = output_dir / "task5_top10_rules_lift.png"
    plt.savefig(p1, dpi=200)
    plt.close()
    print(f"[Visualisasi] Disimpan ke: {p1}")

    # 2. Scatter Support vs Lift
    fig, ax = plt.subplots(figsize=(9, 5.2))
    sc = ax.scatter(
        rules["support"] * 100.0,
        rules["lift"],
        c=rules["confidence"] * 100.0,
        cmap="viridis",
        alpha=0.75,
        edgecolors="black",
        linewidths=0.4,
        s=45,
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Confidence (%)")
    ax.set_title("Hubungan Support vs Lift pada Association Rules (Min Conf 50%)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Support (%)")
    ax.set_ylabel("Lift")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    p2 = output_dir / "task5_support_vs_lift.png"
    plt.savefig(p2, dpi=200)
    plt.close()
    print(f"[Visualisasi] Disimpan ke: {p2}")


# ==============================================================================
# PIPELINE UTAMA ROLE 2
# ==============================================================================
def execute_role2_pipeline(verify_hash: bool = True) -> dict:
    """Menjalankan seluruh pipeline Role 2 (Task 3, 4, 5, 6) pada clean_inproceedings.parquet."""
    t_start = time.perf_counter()
    print("=" * 78)
    print("PIPELINE ROLE 2 — TASK 3, TASK 4, TASK 5, TASK 6 (DBLP 2015-2025)")
    print("=" * 78)

    df, meta_info = load_clean_data(verify_hash=verify_hash)
    venue_name_map, venue_display_map = build_venue_label_map(df)

    # Task 3
    t3 = construct_transactional_databases(df, venue_display_map, OUTPUT_DIR, save_artifacts=True)
    del df
    gc.collect()

    # Task 4 & Task 6
    results_by_sup, benchmark_df = run_fpm_and_benchmark(
        t3["X_author"],
        t3["unique_venues"],
        venue_display_map,
        min_supports=MIN_SUPPORTS,
        min_confidence=MIN_CONFIDENCE,
        output_dir=OUTPUT_DIR,
        save_artifacts=True,
    )

    # Task 5
    t5 = analyze_association_rules(results_by_sup, target_support=0.001, output_dir=OUTPUT_DIR, save_artifacts=True)

    # Simpan ringkasan JSON untuk audit & reproduktibilitas
    summary_payload = {
        "source_parquet_meta": meta_info,
        "task3_stats": t3["stats_raw"],
        "task6_benchmark": benchmark_df.to_dict(orient="records"),
        "task5_total_rules_0_1pct": int(len(t5["rules"])),
        "task5_correlations": t5["corr"],
        "total_pipeline_seconds": round(time.perf_counter() - t_start, 2),
    }
    (OUTPUT_DIR / "role2_summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n[SELESAI] Seluruh pipeline Role 2 selesai dalam {summary_payload['total_pipeline_seconds']} detik.")
    print(f"[OUTPUT] Artefak tersimpan di: {OUTPUT_DIR.resolve()}")
    return summary_payload


if __name__ == "__main__":
    execute_role2_pipeline()
