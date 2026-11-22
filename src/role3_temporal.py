"""
Module src/role3_temporal.py
Implementasi Task 3B (Author-Year Transaction) dan Task E (Temporal Association Analysis)
untuk Anggota 3 (Temporal Analyst & Integration Lead).

Cara pakai:
    python src/role3_temporal.py                 # jalankan penuh
    python src/role3_temporal.py --sample 0.05   # uji cepat 5% data (output ke output/role3_sample)
    python src/role3_temporal.py --verify        # + cek silang dengan pandas murni
"""

import argparse
import gc
import math
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from scipy.sparse import csr_matrix

try:
    from src.data_loader import build_venue_label_map
except ImportError:
    from data_loader import build_venue_label_map

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output" / "role3"
DATA_PATH = ROOT / "data" / "clean_inproceedings.parquet"

PERIODS = [
    {"id": "P1", "label": "2015-2017", "years": (2015, 2017)},
    {"id": "P2", "label": "2018-2020", "years": (2018, 2020)},
    {"id": "P3", "label": "2021-2023", "years": (2021, 2023)},
    {"id": "P4", "label": "2024-2025", "years": (2024, 2025)},
]
PERIOD_IDS = [p["id"] for p in PERIODS]
PERIOD_LABELS = {p["id"]: p["label"] for p in PERIODS}

MIN_CONFIDENCE = 0.25          # 25% untuk Author-Year (71.5% transaksi 1-item; 50% menghapus P2 menjadi 0 rules)
MIN_SUPPORT_TEMPORAL = 0.0005  # 0.05% dari JUMLAH SELURUH transaksi (author-year) per periode
CONF_SENSITIVITY_GRID = [0.25, 0.30, 0.40, 0.50]
MIN_ABS_FLOOR = 5              # batas bawah count absolut
DENSE_BYTES_LIMIT = 1_200_000_000

VENUE_BITS = 16
YEAR_BITS = 4
VENUE_MASK = (1 << VENUE_BITS) - 1
YEAR_MASK = (1 << YEAR_BITS) - 1
AUTHOR_SHIFT = VENUE_BITS + YEAR_BITS

RULE_COLS_EMPTY = [
    "antecedents", "consequents", "antecedent support", "consequent support",
    "support", "confidence", "lift",
]


def log(msg: str):
    print(msg, flush=True)


def format_itemset(itemset, label_map: Dict[str, str]) -> str:
    labels = sorted(label_map.get(k, str(k)) for k in itemset)
    return "{" + ", ".join(labels) + "}"


def format_itemset_keys(itemset) -> str:
    return "{" + ", ".join(sorted(str(k) for k in itemset)) + "}"


# ==============================================================================
# FUNGSI: flatten + bit-packing (author, year, venue)
# ==============================================================================
def _pack_author_year_venue(df_part: pd.DataFrame, year_base: int):
    """
    Mengubah df (1 baris = 1 paper, kolom author_id = list author) menjadi array int64
    berisi tripel UNIK (author, year, venue) yang sudah terurut.

    Layout bit: [ author_code | year_offset (4 bit) | venue_code (16 bit) ]
    Return: packed_unique, unique_venues, unique_authors, n_author_paper_entries
    """
    venue_codes_paper, unique_venues = pd.factorize(df_part["venue_key"], sort=True)
    unique_venues = np.asarray(unique_venues, dtype=object)
    if len(unique_venues) > VENUE_MASK:
        raise ValueError(f"Jumlah venue ({len(unique_venues)}) melebihi kapasitas {VENUE_BITS} bit")

    year_off_paper = df_part["year"].to_numpy(dtype=np.int64) - int(year_base)
    if len(year_off_paper) and (year_off_paper.min() < 0 or year_off_paper.max() > YEAR_MASK):
        raise ValueError("Rentang tahun melebihi kapasitas bit-packing (maks 16 tahun)")

    authors_col = df_part["author_id"].tolist()
    n_auth_per_paper = np.fromiter((len(lst) for lst in authors_col), dtype=np.int64, count=len(authors_col))
    flat_authors = [a for lst in authors_col for a in lst]
    author_codes, unique_authors = pd.factorize(pd.Series(flat_authors, dtype="object"), sort=False)
    n_entries = len(flat_authors)
    del flat_authors, authors_col

    venue_flat = np.repeat(venue_codes_paper.astype(np.int64), n_auth_per_paper)
    year_flat = np.repeat(year_off_paper, n_auth_per_paper)
    packed = (author_codes.astype(np.int64) << AUTHOR_SHIFT) | (year_flat << VENUE_BITS) | venue_flat
    del venue_flat, year_flat, author_codes, venue_codes_paper
    packed = np.unique(packed)
    gc.collect()
    return packed, unique_venues, np.asarray(unique_authors, dtype=object), n_entries


def _group_transactions(packed: np.ndarray):
    """Kelompokkan tripel terurut menjadi transaksi (author, year). Return (keys, start_idx, lens)."""
    ay_keys = packed >> VENUE_BITS
    unique_ay, start_idx, lens = np.unique(ay_keys, return_index=True, return_counts=True)
    return unique_ay, start_idx, lens.astype(np.int64)


# ==============================================================================
# TASK 3B: AUTHOR-YEAR TRANSACTION DATABASE
# ==============================================================================
def _bucket_distribution(lens: np.ndarray) -> pd.DataFrame:
    n = len(lens)
    edges = [("1", lens == 1), ("2", lens == 2), ("3", lens == 3), ("4", lens == 4),
             ("5", lens == 5), ("6-10", (lens >= 6) & (lens <= 10)), (">10", lens > 10)]
    rows = [{"bucket": b, "count": int(m.sum()), "pct": round(float(m.sum()) / n * 100, 2)} for b, m in edges]
    return pd.DataFrame(rows)


def construct_author_year_transactions(
    df: pd.DataFrame,
    venue_display_map: Dict[str, str],
    output_dir: Path = OUTPUT_DIR,
) -> dict:
    log("\n" + "=" * 75)
    log("TASK 3B: KONSTRUKSI AUTHOR-YEAR TRANSACTIONAL DATABASE (Vectorized)")
    log("=" * 75)
    t0 = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    year_min = int(df["year"].min())
    packed, unique_venues, unique_authors, n_entries = _pack_author_year_venue(df, year_min)
    log(f"  [Bit-Packing] {n_entries:,} entri author-paper -> {len(packed):,} tripel (author, year, venue) unik")

    unique_ay, start_idx, tx_lens = _group_transactions(packed)
    n_tx = len(unique_ay)
    n_unique_venues = int(len(np.unique(packed & VENUE_MASK)))
    total_entries = len(packed)

    n_eq1 = int((tx_lens == 1).sum())
    n_ge2 = int((tx_lens >= 2).sum())
    n_ge3 = int((tx_lens >= 3).sum())

    stats_df = pd.DataFrame({
        "Metrik": [
            "Total Transaksi (Author-Year)",
            "Total Item Unik (Venues)",
            "Total Pasangan (Author-Year, Venue)",
            "Rata-rata Venue per Transaksi",
            "Median Venue per Transaksi",
            "Persentil 95 Venue per Transaksi",
            "Persentil 99 Venue per Transaksi",
            "Maksimum Venue per Transaksi",
            "Transaksi 1 Venue (len = 1)",
            "Transaksi Multi-Venue (len >= 2)",
            "Transaksi >= 3 Venue (len >= 3)",
        ],
        "Nilai": [
            f"{n_tx:,}",
            f"{n_unique_venues:,}",
            f"{total_entries:,}",
            f"{tx_lens.mean():.3f}",
            f"{np.median(tx_lens):.1f}",
            f"{np.percentile(tx_lens, 95):.1f}",
            f"{np.percentile(tx_lens, 99):.1f}",
            f"{tx_lens.max():,}",
            f"{n_eq1:,} ({n_eq1 / n_tx * 100:.2f}%)",
            f"{n_ge2:,} ({n_ge2 / n_tx * 100:.2f}%)",
            f"{n_ge3:,} ({n_ge3 / n_tx * 100:.2f}%)",
        ],
    })
    log(stats_df.to_string(index=False))

    # Distribusi panjang transaksi (eksak + bucket) -> bisa dicocokkan dengan Role 2
    exact = np.bincount(tx_lens)
    dist_exact = pd.DataFrame({"jumlah_item": np.arange(len(exact)), "jumlah_transaksi": exact})
    dist_exact = dist_exact[dist_exact["jumlah_transaksi"] > 0].copy()
    dist_exact["persen"] = (dist_exact["jumlah_transaksi"] / n_tx * 100).round(4)
    dist_bucket = _bucket_distribution(tx_lens)
    dist_exact.to_csv(output_dir / "task3b_author_year_length_dist.csv", index=False)
    dist_bucket.to_csv(output_dir / "task3b_author_year_length_bucket.csv", index=False)

    # 10 contoh representatif deterministik
    rng = np.random.default_rng(42)
    buckets = [
        (np.where(tx_lens == 1)[0], 2),
        (np.where(tx_lens == 2)[0], 3),
        (np.where((tx_lens >= 3) & (tx_lens <= 5))[0], 3),
        (np.where(tx_lens >= 6)[0], 2),
    ]
    chosen: List[int] = []
    for pool, k in buckets:
        if len(pool) > 0:
            chosen.extend(rng.choice(pool, size=min(k, len(pool)), replace=False).tolist())
    sample_idx = sorted(chosen[:10], key=lambda i: (-int(tx_lens[i]), i))

    examples = []
    for rank, tx_i in enumerate(sample_idx, start=1):
        s, ln = int(start_idx[tx_i]), int(tx_lens[tx_i])
        a_code = int(unique_ay[tx_i] >> YEAR_BITS)
        yr = year_min + int(unique_ay[tx_i] & YEAR_MASK)
        v_keys = [str(unique_venues[j]) for j in (packed[s:s + ln] & VENUE_MASK)]
        examples.append({
            "No": rank,
            "Author ID": str(unique_authors[a_code]),
            "Tahun": yr,
            "Jumlah Item": len(v_keys),
            "Daftar Konferensi (Names)": ", ".join(venue_display_map.get(k, k) for k in v_keys),
            "Daftar Konferensi (Keys)": ", ".join(v_keys),
        })
    df_examples = pd.DataFrame(examples)
    df_examples.to_csv(output_dir / "task3b_author_year_examples.csv", index=False)
    stats_df.to_csv(output_dir / "task3b_author_year_stats.csv", index=False)

    log(f"  [Task 3B] Selesai dalam {time.perf_counter() - t0:.2f} detik. Berkas CSV tersimpan.")
    result = {
        "stats_df": stats_df, "examples_df": df_examples,
        "dist_exact": dist_exact, "dist_bucket": dist_bucket,
        "n_tx": n_tx, "n_unique_venues": n_unique_venues,
    }
    del packed, unique_ay, start_idx, tx_lens
    gc.collect()
    return result


# ==============================================================================
# TASK E - MINING SATU PERIODE
# ==============================================================================
def _empty_rules() -> pd.DataFrame:
    return pd.DataFrame(columns=RULE_COLS_EMPTY)

def _make_rules(fi: pd.DataFrame, min_conf: float = MIN_CONFIDENCE) -> pd.DataFrame:
    if fi.empty or not (fi["length"] >= 2).any():
        return _empty_rules()
    return association_rules(fi[["support", "itemsets"]], metric="confidence", min_threshold=min_conf)

def _mine_period(df: pd.DataFrame, period: dict, venue_display_map: Dict[str, str], output_dir: Path) -> dict:
    p_id, p_label = period["id"], period["label"]
    y0, y1 = period["years"]
    t_p0 = time.perf_counter()
    log(f"\n>>> Mengeksekusi Periode {p_id}: {p_label} ...")

    df_p = df[(df["year"] >= y0) & (df["year"] <= y1)]
    n_papers = len(df_p)
    result = {
        "id": p_id, "label": p_label, "years": (y0, y1), "n_papers": n_papers,
        "n_tx": 0, "n_multi": 0, "n_itemsets": 0, "n_rules": 0,
        "itemsets": pd.DataFrame(), "rules": pd.DataFrame(), "top10_lift": pd.DataFrame(),
        "venue_tx_count": pd.Series(dtype="int64"),
    }
    if n_papers == 0:
        log(f"  [{p_id}] Tidak ada data pada periode ini.")
        return result

    packed, u_venues, _, _ = _pack_author_year_venue(df_p, y0)
    _, start_idx, lens = _group_transactions(packed)
    n_tx = len(lens)
    venue_codes = packed & VENUE_MASK

    # --- Count venue SEBENARNYA pada seluruh transaksi (untuk support 1-itemset)
    all_v, all_cnt = np.unique(venue_codes, return_counts=True)
    venue_tx_count = pd.Series(all_cnt, index=[str(u_venues[v]) for v in all_v], dtype="int64")

    # --- Transaksi multi-venue
    multi_tx = lens >= 2
    n_multi = int(multi_tx.sum())
    result.update(n_tx=n_tx, n_multi=n_multi, venue_tx_count=venue_tx_count)
    log(f"  [{p_id}] Transaksi Author-Year: {n_tx:,} | Multi-Venue (>=2): {n_multi:,} ({n_multi / n_tx * 100:.1f}%)")
    if n_multi == 0:
        log(f"  [{p_id}] Tidak ada transaksi multi-item.")
        return result

    tx_of_triple = np.repeat(np.arange(n_tx, dtype=np.int64), lens)
    in_multi = multi_tx[tx_of_triple]
    v_multi = venue_codes[in_multi]
    row_multi = (np.cumsum(multi_tx) - 1)[tx_of_triple[in_multi]]

    min_abs_count = max(math.ceil(MIN_SUPPORT_TEMPORAL * n_tx - 1e-12), MIN_ABS_FLOOR)
    v_arr, v_cnt = np.unique(v_multi, return_counts=True)
    freq_codes = v_arr[v_cnt >= min_abs_count]
    if len(freq_codes) < 2:
        min_abs_count = MIN_ABS_FLOOR
        freq_codes = v_arr[v_cnt >= min_abs_count]

    # --- 1-itemset EKSAK dari seluruh transaksi
    single_mask = (venue_tx_count >= min_abs_count).to_numpy()
    singles = venue_tx_count[single_mask]
    fi_single = pd.DataFrame({
        "support": singles.to_numpy() / float(n_tx),
        "count": singles.to_numpy().astype(int),
        "itemsets": [frozenset([k]) for k in singles.index],
    })

    # --- Itemset >= 2 dari transaksi multi-venue
    fi_multi = pd.DataFrame(columns=["support", "count", "itemsets"])
    if len(freq_codes) >= 2:
        col_of_code = np.full(len(u_venues), -1, dtype=np.int64)
        col_of_code[freq_codes] = np.arange(len(freq_codes))
        cols_all = col_of_code[v_multi]
        keep = cols_all >= 0
        col_names = [str(u_venues[c]) for c in freq_codes]
        X = csr_matrix(
            (np.ones(int(keep.sum()), dtype=bool), (row_multi[keep], cols_all[keep])),
            shape=(n_multi, len(freq_codes)), dtype=bool,
        )
        adj_min_support = (min_abs_count - 0.25) / float(n_multi)
        if n_multi * len(freq_codes) <= DENSE_BYTES_LIMIT:
            df_active = pd.DataFrame(X.toarray(), columns=col_names, dtype=bool)
        else:
            df_active = pd.DataFrame.sparse.from_spmatrix(X, columns=col_names)
        fi_raw = fpgrowth(df_active, min_support=adj_min_support, use_colnames=True)
        del df_active, X
        if not fi_raw.empty:
            fi_raw["length"] = fi_raw["itemsets"].apply(len)
            fi_raw = fi_raw[fi_raw["length"] >= 2].copy()
            fi_raw["count"] = np.rint(fi_raw["support"] * n_multi).astype(int)
            fi_raw = fi_raw[fi_raw["count"] >= min_abs_count]
            fi_raw["support"] = fi_raw["count"] / float(n_tx)
            fi_multi = fi_raw[["support", "count", "itemsets"]]
    del packed, venue_codes, tx_of_triple, in_multi, v_multi, row_multi
    gc.collect()

    fi = pd.concat([fi_single, fi_multi], ignore_index=True)
    if fi.empty:
        log(f"  [{p_id}] Tidak ada frequent itemset yang lolos threshold.")
        return result
    fi["length"] = fi["itemsets"].apply(len).astype(int)
    fi["itemsets_keys"] = fi["itemsets"].apply(format_itemset_keys)
    fi["itemsets_names"] = fi["itemsets"].apply(lambda s: format_itemset(s, venue_display_map))
    fi = fi.sort_values(by=["support", "length", "itemsets_keys"], ascending=[False, True, True]).reset_index(drop=True)
    fi_export = fi[["length", "count", "support", "itemsets_keys", "itemsets_names"]]
    fi_export.to_csv(output_dir / f"taskE_itemsets_{p_id.lower()}_{y0}_{y1}.csv", index=False)

    # --- Association rules
    rules = _make_rules(fi, min_conf=MIN_CONFIDENCE)
    csv_name = f"taskE_rules_{p_id.lower()}_{y0}_{y1}.csv"
    if not rules.empty:
        rules["rule_count"] = np.rint(rules["support"] * n_tx).astype(int)
        rules["period"] = p_id
        rules["period_label"] = p_label
        rules["antecedents_keys"] = rules["antecedents"].apply(format_itemset_keys)
        rules["consequents_keys"] = rules["consequents"].apply(format_itemset_keys)
        rules["antecedents_names"] = rules["antecedents"].apply(lambda s: format_itemset(s, venue_display_map))
        rules["consequents_names"] = rules["consequents"].apply(lambda s: format_itemset(s, venue_display_map))
        rules["rule_keys"] = rules["antecedents_keys"] + " -> " + rules["consequents_keys"]
        rules["rule_names"] = rules["antecedents_names"] + " -> " + rules["consequents_names"]
        rules = rules.sort_values(by=["lift", "confidence", "support"], ascending=[False, False, False]).reset_index(drop=True)
        rules.drop(columns=["antecedents", "consequents"]).to_csv(output_dir / csv_name, index=False)
        log(f"  [{p_id}] {len(fi):,} frequent itemsets | {len(rules):,} rules (conf>={MIN_CONFIDENCE*100:.0f}%) -> {csv_name}")
    else:
        pd.DataFrame(columns=[
            "antecedent support", "consequent support", "support", "confidence", "lift",
            "rule_count", "period", "period_label", "rule_keys", "rule_names",
        ]).to_csv(output_dir / csv_name, index=False)
        log(f"  [{p_id}] {len(fi):,} frequent itemsets | 0 rules (confidence >= {MIN_CONFIDENCE*100:.0f}%).")

    result.update(
        n_itemsets=len(fi), n_rules=len(rules), itemsets=fi, rules=rules,
        top10_lift=rules.head(10) if not rules.empty else pd.DataFrame(),
        min_abs_count=min_abs_count,
        n_len1=int((fi["length"] == 1).sum()), n_len2=int((fi["length"] == 2).sum()),
        n_len3=int((fi["length"] == 3).sum()), n_len4p=int((fi["length"] >= 4).sum()),
        runtime_s=round(time.perf_counter() - t_p0, 2),
    )
    log(f"  [{p_id}] Selesai dalam {result['runtime_s']:.2f} detik.")
    return result


# ==============================================================================
# TASK E - PERBANDINGAN ANTARPERIODE
# ==============================================================================
def _classify(row: pd.Series) -> str:
    present = [bool(row[f"lift_{p}"] == row[f"lift_{p}"]) for p in PERIOD_IDS]  # not NaN
    n = sum(present)
    if n == len(PERIOD_IDS):
        return "stable_all"
    if n == len(PERIOD_IDS) - 1:
        return "stable_3"
    if not present[0] and present[-1]:
        return "emerging"
    if present[0] and not present[-1]:
        return "fading"
    return "transient"


def _cross_period_analysis(period_results: dict, venue_display_map: Dict[str, str], output_dir: Path) -> dict:
    log("\n" + "=" * 75)
    log("SINTESIS LINTAS PERIODE")
    log("=" * 75)
    out: dict = {}

    # Ringkasan per periode
    rows = []
    for p_id in PERIOD_IDS:
        r = period_results[p_id]
        if r["n_tx"] == 0:
            continue
        rows.append({
            "period": p_id, "label": r["label"], "n_papers": r["n_papers"], "n_tx": r["n_tx"],
            "n_multi": r["n_multi"], "pct_multi": round(r["n_multi"] / r["n_tx"] * 100, 2),
            "min_abs_count": r.get("min_abs_count"),
            "eff_min_support_pct": round(r.get("min_abs_count", 0) / r["n_tx"] * 100, 4) if r.get("min_abs_count") else None,
            "n_itemsets": r["n_itemsets"], "n_1": r.get("n_len1"), "n_2": r.get("n_len2"),
            "n_3": r.get("n_len3"), "n_4plus": r.get("n_len4p"), "n_rules": r["n_rules"],
            "runtime_s": r.get("runtime_s"),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "taskE_period_summary.csv", index=False)
    out["period_summary"] = summary
    log(summary.to_string(index=False))

    # Dukungan venue per periode (eksak, dari seluruh transaksi)
    sup = pd.DataFrame({p: period_results[p]["venue_tx_count"] / period_results[p]["n_tx"] * 100
                        for p in PERIOD_IDS if period_results[p]["n_tx"] > 0})
    cnt = pd.DataFrame({p: period_results[p]["venue_tx_count"] for p in PERIOD_IDS if period_results[p]["n_tx"] > 0})
    sup = sup.fillna(0.0)
    venue_sup = sup.add_prefix("support_pct_").join(cnt.fillna(0).astype(int).add_prefix("count_"))
    first_p, last_p = sup.columns[0], sup.columns[-1]
    venue_sup["delta_pp_last_vs_first"] = sup[last_p] - sup[first_p]
    venue_sup["venue_name"] = [venue_display_map.get(k, k) for k in venue_sup.index]
    venue_sup = venue_sup.sort_values(f"support_pct_{last_p}", ascending=False)
    venue_sup.index.name = "venue_key"
    venue_sup.to_csv(output_dir / "taskE_venue_support_by_period.csv")
    out["venue_support"] = venue_sup

    all_rules = [period_results[p]["rules"] for p in PERIOD_IDS if not period_results[p]["rules"].empty]
    if not all_rules:
        log("Tidak ada rules pada periode manapun.")
        return out
    df_all = pd.concat(all_rules, ignore_index=True)

    # Matriks kehadiran rule
    lift_piv = df_all.pivot_table(index="rule_keys", columns="period", values="lift", aggfunc="max").reindex(columns=PERIOD_IDS)
    conf_piv = df_all.pivot_table(index="rule_keys", columns="period", values="confidence", aggfunc="max").reindex(columns=PERIOD_IDS)
    sup_piv = df_all.pivot_table(index="rule_keys", columns="period", values="support", aggfunc="max").reindex(columns=PERIOD_IDS)
    names = df_all.drop_duplicates("rule_keys").set_index("rule_keys")["rule_names"]

    pres = lift_piv.notna()
    table = lift_piv.add_prefix("lift_").join(conf_piv.add_prefix("conf_")).join(sup_piv.add_prefix("support_"))
    table.insert(0, "rule_names", names.reindex(table.index))
    table["n_periods"] = pres.sum(axis=1)
    table["first_period"] = pres.idxmax(axis=1)
    table["last_period"] = pres.iloc[:, ::-1].idxmax(axis=1)
    table["mean_lift"] = lift_piv.mean(axis=1)
    table["category"] = table.apply(_classify, axis=1)
    table = table.sort_values(["n_periods", "mean_lift"], ascending=[False, False])
    table.index.name = "rule_keys"
    pres = pres.loc[table.index]  # selaraskan urutan dengan table (sudah di-sort)
    table.to_csv(output_dir / "taskE_rule_presence.csv")
    out["rule_presence"] = table
    log("\nJumlah rule per kategori:\n" + table["category"].value_counts().to_string())

    # Overlap Jaccard antarperiode
    jac = pd.DataFrame(index=PERIOD_IDS, columns=PERIOD_IDS, dtype=float)
    for a in PERIOD_IDS:
        for b in PERIOD_IDS:
            sa, sb = set(pres.index[pres[a]]), set(pres.index[pres[b]])
            jac.loc[a, b] = len(sa & sb) / len(sa | sb) if (sa | sb) else np.nan
    jac.round(4).to_csv(output_dir / "taskE_rule_overlap_jaccard.csv")
    out["jaccard"] = jac

    # 1. Pola bertahan (kompatibel dengan notebook lama)
    stable = table[table["n_periods"] >= 3].copy()
    stable_out = stable[["rule_names"] + [f"lift_{p}" for p in PERIOD_IDS] + ["n_periods", "mean_lift"]].copy()
    stable_out = stable_out.rename(columns={f"lift_{p}": PERIOD_LABELS[p] for p in PERIOD_IDS}).set_index("rule_names")
    stable_out.to_csv(output_dir / "taskE_stable_patterns_lift.csv")
    out["stable"] = stable_out
    log(f"\nPola bertahan (>= 3 periode): {len(stable_out)} | di semua 4 periode: {(table['n_periods'] == 4).sum()}")
    log(stable_out.head(10).to_string())

    # 2. Pergeseran: rule yang TIDAK ada di P1
    shifting = table[~pres["P1"]].copy()
    first_idx = {k: row["first_period"] for k, row in shifting.iterrows()}
    rows = []
    for k, fp in first_idx.items():
        r = df_all[(df_all["rule_keys"] == k) & (df_all["period"] == fp)].iloc[0]
        rows.append({
            "rule_names": r["rule_names"], "period_label": PERIOD_LABELS[fp],
            "count": int(r["rule_count"]), "support": r["support"], "confidence": r["confidence"],
            "lift": r["lift"], "n_periods": int(shifting.loc[k, "n_periods"]), "category": shifting.loc[k, "category"],
        })
    shifting_df = pd.DataFrame(rows)
    if not shifting_df.empty:
        shifting_df = shifting_df.sort_values("lift", ascending=False).head(30)
    shifting_df.to_csv(output_dir / "taskE_shifting_trends.csv", index=False)
    out["shifting"] = shifting_df
    log(f"\nRule baru setelah P1: {len(shifting)} (ditampilkan top 30 berdasarkan lift)")

    # 3. Rule yang menghilang: ada di P1, tidak ada di P4
    fading = table[pres["P1"] & ~pres[PERIOD_IDS[-1]]].copy()
    fading_out = fading[["rule_names", "lift_P1", "support_P1", "n_periods", "last_period"]].sort_values("lift_P1", ascending=False).head(30)
    fading_out.to_csv(output_dir / "taskE_fading_patterns.csv")
    out["fading"] = fading_out
    log(f"Rule yang ada di P1 tapi hilang di {PERIOD_IDS[-1]}: {len(fading)}")

    # 4. Kemunculan venue di dalam rules
    inv = {}
    for p_id in PERIOD_IDS:
        r = period_results[p_id]["rules"]
        if r.empty:
            inv[p_id] = pd.Series(dtype=int)
            continue
        cnts: Dict[str, int] = {}
        for a, c in zip(r["antecedents"], r["consequents"]):
            for v in set(a) | set(c):
                cnts[v] = cnts.get(v, 0) + 1
        inv[p_id] = pd.Series(cnts, dtype=int)
    inv_df = pd.DataFrame(inv).fillna(0).astype(int).add_prefix("n_rules_")
    inv_df["venue_name"] = [venue_display_map.get(k, k) for k in inv_df.index]
    present_cols = [f"n_rules_{p}" for p in PERIOD_IDS if f"n_rules_{p}" in inv_df.columns]
    inv_df["first_period_in_rules"] = (inv_df[present_cols] > 0).idxmax(axis=1).str.replace("n_rules_", "", regex=False)
    inv_df = inv_df.sort_values(present_cols[-1], ascending=False)
    inv_df.index.name = "venue_key"
    inv_df.to_csv(output_dir / "taskE_venue_emergence.csv")
    out["venue_emergence"] = inv_df

    # 5. Uji sensitivitas confidence (menjelaskan mengapa min_conf=25% dipilih pada Author-Year vs 50%)
    sens_rows = []
    for mc in CONF_SENSITIVITY_GRID:
        rsets = {}
        row_s = {"min_support_pct": round(MIN_SUPPORT_TEMPORAL * 100, 3), "min_confidence_pct": int(round(mc * 100))}
        for pid in PERIOD_IDS:
            fi_p = period_results[pid]["itemsets"]
            rp = _make_rules(fi_p, min_conf=mc)
            s_keys = set(zip(rp["antecedents"], rp["consequents"])) if not rp.empty else set()
            rsets[pid] = s_keys
            row_s[f"rules_{pid}"] = len(s_keys)
        all_u = set().union(*rsets.values())
        row_s["stable_4_of_4"] = len(set.intersection(*rsets.values())) if all_u else 0
        row_s["stable_ge_3_of_4"] = sum(1 for k in all_u if sum(k in rsets[p] for p in PERIOD_IDS) >= 3)
        sens_rows.append(row_s)
    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(output_dir / "taskE_threshold_sensitivity.csv", index=False)
    out["threshold_sensitivity"] = sens_df
    log("\nUji sensitivitas confidence (min_support=0.05%):\n" + sens_df.to_string(index=False))

    # Visualisasi (dengan fallback aman jika stable_out kosong agar grafik selalu terbentuk)
    plot_source = stable_out.head(10)[[PERIOD_LABELS[p] for p in PERIOD_IDS]]
    if plot_source.empty:
        fallback_tbl = table.head(10)[["rule_names"] + [f"lift_{p}" for p in PERIOD_IDS]].copy()
        fallback_tbl = fallback_tbl.rename(columns={f"lift_{p}": PERIOD_LABELS[p] for p in PERIOD_IDS}).set_index("rule_names")
        plot_source = fallback_tbl
    plot_temporal_comparison(plot_source, output_dir / "taskE_temporal_comparison.png")
    plot_venue_support_trend(venue_sup, first_p, last_p, output_dir / "taskE_venue_support_trend.png")
    return out


def plot_temporal_comparison(pivot_df: pd.DataFrame, out_path: Path):
    """Garis lift antarperiode. NaN = rule tidak muncul pada periode itu (garis terputus)."""
    if pivot_df.empty:
        return
    plt.figure(figsize=(12.5, 6.2))
    periods = list(pivot_df.columns)
    xs = np.arange(len(periods))  # posisi numerik agar periode tanpa data tetap tampil di sumbu-x
    for rule_name in pivot_df.index:
        short = rule_name if len(rule_name) <= 45 else rule_name[:42] + "..."
        plt.plot(xs, pivot_df.loc[rule_name].to_numpy(dtype=float), marker="o", linewidth=2.2, markersize=6, label=short)
    plt.xticks(xs, periods, fontsize=10)
    plt.title(
        f"Evolusi Nilai Lift Pola Asosiasi Bertahan pada 4 Periode (2015–2025)\n"
        f"(Author-Year DB | Min Support={MIN_SUPPORT_TEMPORAL*100:.2f}%, Min Confidence={MIN_CONFIDENCE*100:.0f}%)",
        fontsize=11.5, fontweight="bold",
    )
    plt.xlabel("Rentang Periode Waktu", fontsize=10.5, fontweight="bold")
    plt.ylabel("Lift (titik hilang = rule di bawah threshold)", fontsize=10.5, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    log(f"[Visualisasi] Plot perbandingan temporal disimpan ke: {out_path}")


def plot_venue_support_trend(venue_sup: pd.DataFrame, first_p: str, last_p: str, out_path: Path, k: int = 8):
    """Venue dengan kenaikan dan penurunan support terbesar (P-awal vs P-akhir)."""
    d = venue_sup.copy()
    d = d[(d[f"count_{first_p}"] >= 50) | (d[f"count_{last_p}"] >= 50)]
    if d.empty:
        return
    cols = [c for c in venue_sup.columns if c.startswith("support_pct_")]
    labels = [PERIOD_LABELS[c.replace("support_pct_", "")] for c in cols]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for ax, (title, sub) in zip(axes, [("Kenaikan support terbesar", d.nlargest(k, "delta_pp_last_vs_first")),
                                       ("Penurunan support terbesar", d.nsmallest(k, "delta_pp_last_vs_first"))]):
        for key, row in sub.iterrows():
            ax.plot(labels, [row[c] for c in cols], marker="o", linewidth=2, label=row["venue_name"])
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Periode")
        ax.set_ylabel("Support (% transaksi author-year)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)
    fig.suptitle("Perubahan Popularitas Venue Antarperiode (support eksak)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    log(f"[Visualisasi] Plot tren support venue disimpan ke: {out_path}")


# ==============================================================================
# TASK E - ENTRY POINT
# ==============================================================================
def run_temporal_mining(
    df: pd.DataFrame,
    venue_display_map: Dict[str, str],
    output_dir: Path = OUTPUT_DIR,
) -> dict:
    log("\n" + "=" * 75)
    log("TASK E: TEMPORAL ASSOCIATION ANALYSIS (4 PERIODE)")
    log("=" * 75)
    output_dir.mkdir(parents=True, exist_ok=True)

    period_results = {p["id"]: _mine_period(df, p, venue_display_map, output_dir) for p in PERIODS}
    period_results["_cross"] = _cross_period_analysis(period_results, venue_display_map, output_dir)
    return period_results

def main():
    ap = argparse.ArgumentParser(description="Pipeline Anggota 3: Task 3B & Task E")
    ap.add_argument("--sample", type=float, default=None, help="Fraksi data (0-1) untuk uji cepat; output ke output/role3_sample")
    ap.add_argument("--verify", action="store_true", help="Cek silang hasil dengan pandas murni")
    ap.add_argument("--data", type=str, default=str(DATA_PATH), help="Path clean_inproceedings.parquet")
    args = ap.parse_args()

    log("=" * 75)
    log("PIPELINE ANGGOTA 3: TEMPORAL ASSOCIATIONS & TRANSACTION DB B (v2)")
    log("=" * 75)

    data_path = Path(args.data)
    if not data_path.is_file():
        log(f"ERROR: Dataset bersih {data_path} tidak ditemukan!")
        return

    out_dir = OUTPUT_DIR
    t0 = time.perf_counter()
    df = pd.read_parquet(data_path, columns=["author_id", "year", "venue_key", "venue_name"])
    log(f"Dataset dimuat: {len(df):,} publikasi ({time.perf_counter() - t0:.2f}s).")
    if args.sample:
        df = df.sample(frac=args.sample, random_state=42).reset_index(drop=True)
        out_dir = OUTPUT_DIR.parent / "role3_sample"
        log(f"[MODE SAMPLE] {len(df):,} publikasi dipakai. Output ke {out_dir}")

    _, venue_display_map = build_venue_label_map(df)

    if args.verify:
        log("\n[Verify] Cek silang bit-packing vs pandas groupby pada 20.000 baris sampel...")
        df_v = df.head(20_000)
        packed_v, _, _, _ = _pack_author_year_venue(df_v, int(df_v["year"].min()))
        u_ay_v, _, _ = _group_transactions(packed_v)
        exp_v = df_v[["author_id", "year", "venue_key"]].explode("author_id").drop_duplicates()
        n_tx_pd = exp_v.groupby(["author_id", "year"]).ngroups
        assert len(packed_v) == len(exp_v), f"NNZ beda: {len(packed_v)} vs {len(exp_v)}"
        assert len(u_ay_v) == n_tx_pd, f"n_tx beda: {len(u_ay_v)} vs {n_tx_pd}"
        log(f"[Verify] PASS - Bit-packing 100% cocok dengan pandas murni ({n_tx_pd:,} tx, {len(exp_v):,} pasangan unik).")

    t3b = construct_author_year_transactions(df, venue_display_map, out_dir)
    res = run_temporal_mining(df, venue_display_map, out_dir)

    log("\n" + "=" * 75)
    log("SELURUH EKSPERIMEN ANGGOTA 3 (TASK 3B & TASK E) SELESAI")
    log(f"Berkas CSV dan grafik disimpan di: {out_dir.resolve()}")
    log("=" * 75)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    main()