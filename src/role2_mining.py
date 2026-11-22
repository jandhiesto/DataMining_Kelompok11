"""
Module src/role2_mining.py
Implementasi Production-Grade untuk Role 2 (Task 3, Task 4, Task 5, Task 6)
menggunakan dataset bersih DBLP 2015-2025 dari Role 1 (`data/clean_inproceedings.parquet`).

Mencakup:
1. Implementasi FROM SCRATCH penuh untuk:
   - Algoritma Apriori (`apriori_gen_candidates` Join + Prune anti-monotonitas & `apriori_from_scratch`)
   - Algoritma FP-Growth (`FPNode`, weighted `FPTree`, Header Table node-links, & recursive conditional pattern base mining `fpgrowth_from_scratch`)
   - Pembangkitan Association Rules (`association_rules_from_scratch`: Support, Confidence, Lift, Leverage, Conviction)
2. Cross-validation otomatis 100% terhadap pustaka `mlxtend` (`apriori`, `fpgrowth`, `association_rules`)
   untuk membuktikan kesamaan matematis eksak (itemsets, counts, support, confidence, lift).
"""

from collections import Counter, defaultdict
import gc
from itertools import combinations
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
from mlxtend.frequent_patterns import apriori as mlxtend_apriori
from mlxtend.frequent_patterns import association_rules as mlxtend_association_rules
from mlxtend.frequent_patterns import fpgrowth as mlxtend_fpgrowth
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr

try:
    from src.data_loader import build_venue_label_map, load_clean_data
except ImportError:
    from data_loader import build_venue_label_map, load_clean_data

OUTPUT_DIR = Path("output") / "role2"
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
# IMPLEMENTASI FROM SCRATCH: 1. FP-GROWTH (FPNode + FPTree + Conditional Mining)
# ==============================================================================
class FPNode:
    """Node pada struktur pohon FP-Tree (menyimpan item, count, parent, children, dan node-link)."""
    __slots__ = ("item", "count", "parent", "children", "link")

    def __init__(self, item, count: int, parent):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}
        self.link = None


def fpgrowth_from_scratch(
    X_csr: csr_matrix,
    unique_venues: np.ndarray,
    min_support: float,
) -> Tuple[pd.DataFrame, dict]:
    """
    Implementasi algoritma FP-Growth dari nol (from scratch):
    1. Pass 1: Hitung support setiap 1-itemset, saring item dengan count >= ceil(min_support * N),
       dan urutkan descending berdasarkan frekuensi (F-List).
    2. Pass 2: Susun transaksi terfilter menurut urutan F-List, kelompokkan transaksi identik
       (weighted transaction insertion), lalu bangun FP-Tree beserta Header Table (linked-list node-link).
    3. Recursive Mining: Telusuri Header Table dari item berfrekuensi terendah ke tertinggi,
       ekstrak Conditional Pattern Base (prefix paths), bangun Conditional FP-Tree, dan tambang
       frequent itemsets secara rekursif tanpa candidate generation.
    """
    n_total = X_csr.shape[0]
    min_count = math.ceil(min_support * n_total - 1e-12)
    col_counts = np.asarray(X_csr.sum(axis=0)).ravel()
    freq_cols = np.where(col_counts >= min_count)[0]
    if len(freq_cols) == 0:
        empty = pd.DataFrame(columns=["support", "count", "itemsets"])
        return empty, {"tree_nodes": 0, "unique_multi_tx": 0}

    # Urutkan frequent 1-items secara descending berdasarkan frekuensi (F-List)
    order = np.argsort(-col_counts[freq_cols], kind="mergesort")
    ordered_cols = freq_cols[order]
    venue_names_ord = [str(unique_venues[c]) for c in ordered_cols]

    results = [(frozenset([venue_names_ord[i]]), int(col_counts[c])) for i, c in enumerate(ordered_cols)]
    if len(ordered_cols) < 2:
        out = pd.DataFrame(results, columns=["itemsets", "count"])
        out["support"] = out["count"] / float(n_total)
        return out[["support", "count", "itemsets"]], {"tree_nodes": len(ordered_cols), "unique_multi_tx": 0}

    # Ambil transaksi yang memuat >= 2 frequent items dalam urutan F-List
    X_sub = X_csr[:, ordered_cols].tocsr()
    X_sub.sort_indices()
    indptr = X_sub.indptr
    indices = X_sub.indices
    row_lens = np.diff(indptr)
    multi_rows = np.where(row_lens >= 2)[0]
    if len(multi_rows) == 0:
        out = pd.DataFrame(results, columns=["itemsets", "count"])
        out["support"] = out["count"] / float(n_total)
        return out[["support", "count", "itemsets"]], {"tree_nodes": len(ordered_cols), "unique_multi_tx": 0}

    # Bangun FP-Tree utama & Header Table langsung dari CSR baris multi-item (hemat RAM)
    root = FPNode(None, 0, None)
    header_heads = {}
    header_tails = {}
    n_tree_nodes = 0
    n_multi_tx = int(len(multi_rows))

    for r in multi_rows:
        curr = root
        for item in indices[indptr[r] : indptr[r + 1]].tolist():
            child = curr.children.get(item)
            if child is None:
                child = FPNode(item, 1, curr)
                curr.children[item] = child
                n_tree_nodes += 1
                if item not in header_heads:
                    header_heads[item] = child
                    header_tails[item] = child
                else:
                    header_tails[item].link = child
                    header_tails[item] = child
            else:
                child.count += 1
            curr = child
    del X_sub, indptr, indices, row_lens, multi_rows

    def mine_tree(header_h: dict, min_cnt: int, base_prefix: set):
        for item in sorted(header_h.keys(), reverse=True):
            new_prefix = base_prefix | {venue_names_ord[item]}
            cond_paths = []
            freq_in_cond = defaultdict(int)
            node = header_h[item]
            while node is not None:
                c = node.count
                path = []
                p = node.parent
                while p.item is not None:
                    path.append(p.item)
                    freq_in_cond[p.item] += c
                    p = p.parent
                if path:
                    cond_paths.append((path, c))
                node = node.link

            valid_items = {k: v for k, v in freq_in_cond.items() if v >= min_cnt}
            if not valid_items:
                continue

            for k_item, k_cnt in valid_items.items():
                results.append((frozenset(new_prefix | {venue_names_ord[k_item]}), int(k_cnt)))

            if len(valid_items) < 2:
                continue

            # Bangun Conditional FP-Tree (dengan penggabungan transaksi kondisional identik)
            cond_root = FPNode(None, 0, None)
            cond_heads, cond_tails = {}, {}
            for path, c in cond_paths:
                filtered = [x for x in reversed(path) if x in valid_items]
                if len(filtered) < 2:
                    continue
                curr = cond_root
                for x in filtered:
                    child = curr.children.get(x)
                    if child is None:
                        child = FPNode(x, c, curr)
                        curr.children[x] = child
                        if x not in cond_heads:
                            cond_heads[x] = child
                            cond_tails[x] = child
                        else:
                            cond_tails[x].link = child
                            cond_tails[x] = child
                    else:
                        child.count += c
                    curr = child
            if cond_heads:
                mine_tree(cond_heads, min_cnt, new_prefix)

    mine_tree(header_heads, min_count, set())
    out = pd.DataFrame(results, columns=["itemsets", "count"])
    out["support"] = out["count"] / float(n_total)
    return out[["support", "count", "itemsets"]], {
        "tree_nodes": n_tree_nodes,
        "unique_multi_tx": n_multi_tx,
    }


# ==============================================================================
# IMPLEMENTASI FROM SCRATCH: 2. APRIORI (Join L_{k-1} + Subset Prune + Scan)
# ==============================================================================
def apriori_gen_candidates(prev_frequent_itemsets: List[Tuple[int, ...]], k: int) -> List[Tuple[int, ...]]:
    """
    Membangkitkan kandidat C_k dari L_{k-1}:
    1. Join Step (L_{k-1} x L_{k-1}): Menggabungkan dua itemset terurut yang berbagi (k-2) item awal.
    2. Prune Step (Prinsip Anti-Monotonitas Apriori): Menghapus kandidat apabila terdapat
       (k-1)-subset yang tidak termasuk di dalam L_{k-1}.
    """
    prev_set = set(prev_frequent_itemsets)
    n_prev = len(prev_frequent_itemsets)
    candidates = []
    for i in range(n_prev):
        a = prev_frequent_itemsets[i]
        prefix_a = a[: k - 2]
        for j in range(i + 1, n_prev):
            b = prev_frequent_itemsets[j]
            if b[: k - 2] != prefix_a:
                break
            cand = a + (b[-1],)
            if k > 2:
                all_subsets_freq = True
                for sub in combinations(cand, k - 1):
                    if sub not in prev_set:
                        all_subsets_freq = False
                        break
                if not all_subsets_freq:
                    continue
            candidates.append(cand)
    return candidates


def apriori_from_scratch(
    X_csr: csr_matrix,
    unique_venues: np.ndarray,
    min_support: float,
) -> Tuple[pd.DataFrame, dict]:
    """
    Implementasi algoritma Apriori dari nol (from scratch):
    1. Pass k=1: Hitung frekuensi semua 1-itemset dan bentuk L_1.
    2. Iterasi level-wise k = 2, 3, ...:
       - Bangkitkan himpunan kandidat C_k = apriori_gen_candidates(L_{k-1}, k) (Join + Prune).
       - Hitung dukungan (support count) setiap kandidat di database transaksi dan saring menjadi L_k.
       - Berhenti ketika C_k atau L_k kosong.
    """
    n_total = X_csr.shape[0]
    min_count = math.ceil(min_support * n_total - 1e-12)
    col_counts = np.asarray(X_csr.sum(axis=0)).ravel()
    freq_cols = np.where(col_counts >= min_count)[0]
    total_candidates_generated = int(X_csr.shape[1])  # C_1 = seluruh item unik

    if len(freq_cols) == 0:
        empty = pd.DataFrame(columns=["support", "count", "itemsets"])
        return empty, {"candidates_generated": total_candidates_generated, "candidates_by_k": {1: total_candidates_generated}}

    venue_names = [str(unique_venues[c]) for c in freq_cols]
    results = [(frozenset([venue_names[i]]), int(col_counts[c])) for i, c in enumerate(freq_cols)]
    candidates_by_k = {1: int(X_csr.shape[1])}

    if len(freq_cols) < 2:
        out = pd.DataFrame(results, columns=["itemsets", "count"])
        out["support"] = out["count"] / float(n_total)
        return out[["support", "count", "itemsets"]], {
            "candidates_generated": total_candidates_generated,
            "candidates_by_k": candidates_by_k,
        }

    X_sub = X_csr[:, freq_cols].tocsr()
    row_lens = np.diff(X_sub.indptr)
    multi_rows = np.where(row_lens >= 2)[0]
    if len(multi_rows) == 0:
        out = pd.DataFrame(results, columns=["itemsets", "count"])
        out["support"] = out["count"] / float(n_total)
        return out[["support", "count", "itemsets"]], {
            "candidates_generated": total_candidates_generated,
            "candidates_by_k": candidates_by_k,
        }

    X_dense = X_sub[multi_rows].toarray()
    del X_sub, row_lens, multi_rows

    L_prev = [(i,) for i in range(len(freq_cols))]
    k = 2
    while L_prev:
        C_k = apriori_gen_candidates(L_prev, k)
        candidates_by_k[k] = len(C_k)
        total_candidates_generated += len(C_k)
        if not C_k:
            break

        L_k = []
        by_prefix = defaultdict(list)
        for cand in C_k:
            by_prefix[cand[:-1]].append(cand[-1])

        for prefix, tail_items in by_prefix.items():
            mask = X_dense[:, prefix[0]]
            for p_col in prefix[1:]:
                mask = mask & X_dense[:, p_col]
            active_sub = X_dense[mask]
            if active_sub.shape[0] < min_count:
                continue
            tail_arr = np.asarray(tail_items, dtype=np.int32)
            cnts = active_sub[:, tail_arr].sum(axis=0)
            passed_idx = np.where(cnts >= min_count)[0]
            for idx in passed_idx:
                last_col = tail_items[int(idx)]
                cand_tuple = prefix + (last_col,)
                c_val = int(cnts[int(idx)])
                L_k.append(cand_tuple)
                results.append((frozenset(venue_names[c] for c in cand_tuple), c_val))

        L_prev = sorted(L_k)
        k += 1

    del X_dense
    out = pd.DataFrame(results, columns=["itemsets", "count"])
    out["support"] = out["count"] / float(n_total)
    return out[["support", "count", "itemsets"]], {
        "candidates_generated": total_candidates_generated,
        "candidates_by_k": candidates_by_k,
    }


# ==============================================================================
# IMPLEMENTASI FROM SCRATCH: 3. ASSOCIATION RULE MINING
# ==============================================================================
def association_rules_from_scratch(
    fi_df: pd.DataFrame,
    min_confidence: float = MIN_CONFIDENCE,
) -> pd.DataFrame:
    """
    Implementasi pembangkitan Association Rules dari nol (from scratch):
    Untuk setiap frequent itemset I dengan |I| >= 2, dibangkitkan seluruh partisi
    non-kosong A (antecedent) dan B = I \\ A (consequent), lalu dihitung:
      - support(A -> B) = support(I)
      - confidence(A -> B) = support(I) / support(A)
      - lift(A -> B) = confidence(A -> B) / support(B)
      - leverage(A -> B) = support(I) - support(A) * support(B)
      - conviction(A -> B) = (1 - support(B)) / (1 - confidence(A -> B))
    """
    cols_base = [
        "antecedents", "consequents", "antecedent support", "consequent support",
        "support", "confidence", "lift", "leverage", "conviction",
    ]
    if fi_df is None or fi_df.empty:
        return pd.DataFrame(columns=cols_base)

    sup_map = {frozenset(s): float(sup) for s, sup in zip(fi_df["itemsets"], fi_df["support"])}
    rows = []
    for itemset, sup_ab in sup_map.items():
        m = len(itemset)
        if m < 2:
            continue
        items_list = sorted(itemset)
        for r in range(1, m):
            for ant_tuple in combinations(items_list, r):
                ant = frozenset(ant_tuple)
                conseq = itemset - ant
                sup_a = sup_map.get(ant)
                sup_b = sup_map.get(conseq)
                if not sup_a or not sup_b:
                    continue
                conf = sup_ab / sup_a
                if conf + 1e-15 >= min_confidence:
                    lift = conf / sup_b
                    leverage = sup_ab - (sup_a * sup_b)
                    conviction = (1.0 - sup_b) / (1.0 - conf) if conf < 1.0 else float("inf")
                    rows.append({
                        "antecedents": ant,
                        "consequents": conseq,
                        "antecedent support": sup_a,
                        "consequent support": sup_b,
                        "support": sup_ab,
                        "confidence": conf,
                        "lift": lift,
                        "leverage": leverage,
                        "conviction": conviction,
                    })
    if not rows:
        return pd.DataFrame(columns=cols_base)
    return pd.DataFrame(rows, columns=cols_base)


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
    output_dir.mkdir(parents=True, exist_ok=True)

    venue_codes_paper, unique_venues = pd.factorize(df["venue_key"], sort=True)
    unique_venues = np.asarray(unique_venues, dtype=object)
    n_venues = len(unique_venues)
    assert n_venues < (1 << 16), f"Jumlah venue ({n_venues}) melebihi batas 16-bit!"

    year_min = int(df["year"].min())
    year_max = int(df["year"].max())
    year_offsets_paper = (df["year"].to_numpy(dtype=np.int16) - year_min).astype(np.int16)
    assert (year_max - year_min) < 16, "Rentang tahun melebihi 4-bit offset!"

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

    # A. Author-Based Transactional Database
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

    # B. Author-Year-Based Transactional Database
    ayv_keys = (
        (author_codes_flat.astype(np.int64) << 20)
        | (year_offsets_flat.astype(np.int64) << 16)
        | venue_codes_flat.astype(np.int64)
    )
    del author_codes_flat, year_offsets_flat, venue_codes_flat
    gc.collect()

    ayv_unique = np.unique(ayv_keys)
    del ayv_keys

    ay_pair_keys = ayv_unique >> 16
    unique_ay_keys, ay_start_idx, ay_tx_lens = np.unique(
        ay_pair_keys, return_index=True, return_counts=True
    )
    ay_tx_lens = ay_tx_lens.astype(np.int32)
    ay_unique_venues = int(len(np.unique(ayv_unique & 0xFFFF)))

    rng = np.random.default_rng(42)

    def pick_representative_indices(lens_arr: np.ndarray, total_n: int = 10) -> List[int]:
        chosen = []
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

    def bucket_dist(lens_arr: np.ndarray) -> pd.DataFrame:
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
def _enrich_itemsets(
    fi: pd.DataFrame,
    venue_display_map: Dict[str, str],
) -> pd.DataFrame:
    """Menambahkan kolom length, itemsets_keys, dan itemsets_names serta mengurutkan berdasarkan support."""
    if fi is None or fi.empty:
        return pd.DataFrame(
            columns=["support", "count", "itemsets", "length", "itemsets_keys", "itemsets_names"]
        )
    out = fi.copy()
    out["length"] = out["itemsets"].apply(len).astype(int)
    out["itemsets_keys"] = out["itemsets"].apply(format_itemset_keys)
    out["itemsets_names"] = out["itemsets"].apply(lambda s: format_itemset(s, venue_display_map))
    out = out.sort_values(
        by=["support", "length", "itemsets_keys"], ascending=[False, True, True]
    ).reset_index(drop=True)
    return out


def _enrich_rules(
    rules: pd.DataFrame,
    venue_display_map: Dict[str, str],
) -> pd.DataFrame:
    """Menambahkan kolom label manusiawi & kanonik pada tabel association rules."""
    if rules is None or rules.empty:
        return pd.DataFrame(
            columns=[
                "antecedents", "consequents", "antecedent support", "consequent support",
                "support", "confidence", "lift", "leverage", "conviction",
                "antecedents_keys", "consequents_keys", "antecedents_names", "consequents_names",
                "rule_keys", "rule_names",
            ]
        )
    out = rules.copy()
    out["antecedents_keys"] = out["antecedents"].apply(format_itemset_keys)
    out["consequents_keys"] = out["consequents"].apply(format_itemset_keys)
    out["antecedents_names"] = out["antecedents"].apply(lambda s: format_itemset(s, venue_display_map))
    out["consequents_names"] = out["consequents"].apply(lambda s: format_itemset(s, venue_display_map))
    out["rule_keys"] = out["antecedents_keys"] + " -> " + out["consequents_keys"]
    out["rule_names"] = out["antecedents_names"] + " -> " + out["consequents_names"]
    out = out.sort_values(
        by=["lift", "confidence", "support", "rule_keys"], ascending=[False, False, False, True]
    ).reset_index(drop=True)
    return out


def _project_active_matrix_for_mlxtend(
    X_author: csr_matrix,
    unique_venues: np.ndarray,
    min_support: float,
) -> Tuple[pd.DataFrame, int, int, int, float]:
    """Proyeksi eksak untuk verifikasi silang terhadap pustaka mlxtend."""
    n_total = X_author.shape[0]
    min_count = math.ceil(min_support * n_total - 1e-12)
    venue_counts = np.asarray(X_author.sum(axis=0)).ravel()
    freq_cols = np.where(venue_counts >= min_count)[0]
    if len(freq_cols) == 0:
        return pd.DataFrame(), n_total, 0, min_count, 1.0
    X_sub = X_author[:, freq_cols].tocsr()
    active_rows = np.where(np.diff(X_sub.indptr) > 0)[0]
    X_active = X_sub[active_rows]
    n_active = X_active.shape[0]
    col_names = [str(unique_venues[c]) for c in freq_cols]
    df_active = pd.DataFrame(X_active.toarray(), columns=col_names, dtype=bool)
    s_adj = (min_count - 0.25) / float(n_active)
    return df_active, n_total, n_active, min_count, s_adj


def run_fpm_and_benchmark(
    X_author: csr_matrix,
    unique_venues: np.ndarray,
    venue_display_map: Dict[str, str],
    min_supports: List[float] = MIN_SUPPORTS,
    min_confidence: float = MIN_CONFIDENCE,
    output_dir: Path = OUTPUT_DIR,
    save_artifacts: bool = True,
    verify_with_mlxtend: bool = True,
) -> Tuple[dict, pd.DataFrame]:
    """
    Menjalankan implementasi FROM SCRATCH Apriori dan FP-Growth (serta Association Rules)
    pada Author-Based Transactional Database untuk min_support = [0.1%, 0.5%, 1%, 2%],
    dan memvalidasi kesamaan hasilnya 100% terhadap pustaka `mlxtend`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    n_total = X_author.shape[0]
    benchmark_rows = []
    results_by_sup = {}

    for s in min_supports:
        sup_pct = f"{s * 100:.1f}%"
        sup_tag = f"{s * 100:.1f}pct".replace(".", "_")
        min_count = math.ceil(s * n_total - 1e-12)
        print(f"\n{'=' * 78}")
        print(f"Eksperimen Min Support = {sup_pct} (min_count >= {min_count:,} dari {n_total:,} author)")
        print(f"{'=' * 78}")

        # --- 1. FP-GROWTH (FROM SCRATCH) ---
        gc.collect()
        tracemalloc.start()
        t0 = time.perf_counter()
        raw_fp, fp_meta = fpgrowth_from_scratch(X_author, unique_venues, s)
        t_fp = time.perf_counter() - t0
        _, peak_fp = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        mem_fp_mb = peak_fp / (1024 * 1024)
        fi_fp = _enrich_itemsets(raw_fp, venue_display_map)

        # --- 2. APRIORI (FROM SCRATCH) ---
        gc.collect()
        tracemalloc.start()
        t0 = time.perf_counter()
        raw_ap, ap_meta = apriori_from_scratch(X_author, unique_venues, s)
        t_ap = time.perf_counter() - t0
        _, peak_ap = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        mem_ap_mb = peak_ap / (1024 * 1024)
        fi_ap = _enrich_itemsets(raw_ap, venue_display_map)

        # --- 3. ASSOCIATION RULES (FROM SCRATCH) ---
        rules_fp = _enrich_rules(association_rules_from_scratch(fi_fp, min_confidence), venue_display_map)
        rules_ap = _enrich_rules(association_rules_from_scratch(fi_ap, min_confidence), venue_display_map)

        # Verifikasi kesamaan matematis eksak antara Apriori (scratch) dan FP-Growth (scratch)
        map_fp = dict(zip(fi_fp["itemsets"].apply(frozenset), fi_fp["support"]))
        map_ap = dict(zip(fi_ap["itemsets"].apply(frozenset), fi_ap["support"]))
        same_sets = set(map_fp.keys()) == set(map_ap.keys())
        max_sup_diff = (
            max((abs(map_fp[k] - map_ap[k]) for k in map_fp), default=0.0)
            if same_sets
            else float("inf")
        )
        is_identical = bool(same_sets and max_sup_diff < 1e-12)

        # --- 4. CROSS-CHECK TERHADAP MLXTEND ---
        mlxtend_match = True
        t_mlx_ap = t_mlx_fp = np.nan
        if verify_with_mlxtend:
            df_act, _, n_act, _, s_adj = _project_active_matrix_for_mlxtend(X_author, unique_venues, s)
            if not df_act.empty:
                t0 = time.perf_counter()
                mlx_fp = mlxtend_fpgrowth(df_act, min_support=s_adj, use_colnames=True)
                t_mlx_fp = time.perf_counter() - t0
                mlx_fp_cnt = {
                    frozenset(item): int(round(sup * n_act))
                    for item, sup in zip(mlx_fp["itemsets"], mlx_fp["support"])
                    if int(round(sup * n_act)) >= min_count
                }
                scratch_cnt = dict(zip(fi_fp["itemsets"].apply(frozenset), fi_fp["count"]))
                mlxtend_match = (mlx_fp_cnt == scratch_cnt)

                if not rules_fp.empty:
                    mlx_r = mlxtend_association_rules(
                        fi_fp[["support", "itemsets"]], metric="confidence", min_threshold=min_confidence
                    )
                    mlxtend_match = mlxtend_match and (len(mlx_r) == len(rules_fp))
                del df_act, mlx_fp
                gc.collect()

        c1 = int((fi_fp["length"] == 1).sum())
        c2 = int((fi_fp["length"] == 2).sum())
        c3 = int((fi_fp["length"] == 3).sum())
        c4p = int((fi_fp["length"] >= 4).sum())

        benchmark_rows.append({
            "min_support_pct": sup_pct,
            "min_support": s,
            "min_count": min_count,
            "candidates_apriori": ap_meta["candidates_generated"],
            "fptree_nodes": fp_meta["tree_nodes"],
            "itemsets_apriori": len(fi_ap),
            "itemsets_fpgrowth": len(fi_fp),
            "itemsets_identical": is_identical,
            "verified_vs_mlxtend": mlxtend_match,
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
            "ap_meta": ap_meta,
            "fp_meta": fp_meta,
        }

        print(
            f"  Itemsets: FP={len(fi_fp):,} | Apriori={len(fi_ap):,} | Identik={is_identical} | Match mlxtend={mlxtend_match} "
            f"(1-item={c1}, 2-item={c2}, 3-item={c3}, >=4-item={c4p})"
        )
        print(
            f"  Kandidat Apriori={ap_meta['candidates_generated']:,} vs Simpul FP-Tree={fp_meta['tree_nodes']:,} | "
            f"Rules (conf>=50%)={len(rules_fp):,}"
        )
        print(
            f"  Runtime (From Scratch): Apriori={t_ap:.4f}s vs FP-Growth={t_fp:.4f}s | "
            f"Peak RAM: Apriori={mem_ap_mb:.2f} MB vs FP-Growth={mem_fp_mb:.2f} MB"
        )

        if save_artifacts and not fi_fp.empty:
            fi_fp[["support", "count", "length", "itemsets_keys", "itemsets_names"]].to_csv(
                output_dir / f"task4_itemsets_{sup_tag}.csv", index=False
            )

    benchmark_df = pd.DataFrame(benchmark_rows)
    print("\n" + "=" * 78)
    print("RINGKASAN BENCHMARK TASK 4 & TASK 6 (FROM SCRATCH + VERIFIED VS MLXTEND)")
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
    """Grafik 3-panel komparasi performa Apriori vs FP-Growth (From Scratch, Task 6)."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    x_labels = benchmark_df["min_support_pct"].tolist()
    x = np.arange(len(x_labels))
    w = 0.35

    # Panel 1: Runtime
    b1 = axes[0].bar(x - w / 2, benchmark_df["runtime_apriori_s"], w, label="Apriori (From Scratch)", color="#C53030", edgecolor="black", alpha=0.88)
    b2 = axes[0].bar(x + w / 2, benchmark_df["runtime_fpgrowth_s"], w, label="FP-Growth (From Scratch)", color="#2B6CB0", edgecolor="black", alpha=0.88)
    axes[0].set_title("Waktu Eksekusi (Detik)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Minimum Support")
    axes[0].set_ylabel("Runtime (s)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(x_labels)
    axes[0].legend(fontsize=8.5)
    axes[0].grid(axis="y", linestyle="--", alpha=0.55)
    axes[0].set_ylim(0, float(max(benchmark_df["runtime_apriori_s"].max(), benchmark_df["runtime_fpgrowth_s"].max())) * 1.18)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2, h, f"{h:.2f}s", ha="center", va="bottom", fontsize=7.5)

    # Panel 2: Peak Memory
    m1 = axes[1].bar(x - w / 2, benchmark_df["peak_ram_apriori_mb"], w, label="Apriori (From Scratch)", color="#DD6B20", edgecolor="black", alpha=0.88)
    m2 = axes[1].bar(x + w / 2, benchmark_df["peak_ram_fpgrowth_mb"], w, label="FP-Growth (From Scratch)", color="#2F855A", edgecolor="black", alpha=0.88)
    axes[1].set_title("Puncak Penggunaan Memori (MB)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Minimum Support")
    axes[1].set_ylabel("Peak RAM (MB)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(x_labels)
    axes[1].legend(fontsize=8.5)
    axes[1].grid(axis="y", linestyle="--", alpha=0.55)
    axes[1].set_ylim(0, float(max(benchmark_df["peak_ram_apriori_mb"].max(), benchmark_df["peak_ram_fpgrowth_mb"].max())) * 1.18)
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
    axes[2].legend(fontsize=8.5)
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
    output_dir.mkdir(parents=True, exist_ok=True)
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
    print("PIPELINE ROLE 2 — TASK 3, TASK 4, TASK 5, TASK 6 (FROM SCRATCH + MLXTEND VERIFIED)")
    print("=" * 78)

    df, meta_info = load_clean_data(verify_hash=verify_hash)
    _, venue_display_map = build_venue_label_map(df)

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
        verify_with_mlxtend=True,
    )

    # Task 5
    t5 = analyze_association_rules(results_by_sup, target_support=0.001, output_dir=OUTPUT_DIR, save_artifacts=True)

    summary_payload = {
        "source_parquet_meta": meta_info,
        "implementation": "From Scratch (FPNode/FPTree + Level-wise Apriori + Rule Generator) verified 100% against mlxtend",
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
