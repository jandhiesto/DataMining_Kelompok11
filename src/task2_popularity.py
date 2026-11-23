"""
Module src/task2_popularity.py
Implementasi Task 2 Conference Popularity Analysis menggunakan dataset bersih
DBLP 2015-2025 (`data/clean_inproceedings.parquet`).

Cakupan sesuai spesifikasi proyek (Task 2):
1. Identifikasi 10 konferensi dengan jumlah publikasi terbanyak.
2. Identifikasi 10 konferensi dengan jumlah author unik terbanyak.
3. Analisis tren jumlah publikasi konferensi berdasarkan tahun (2015-2025).
4. Perbandingan ranking berdasarkan jumlah publikasi vs jumlah author unik
   (Spearman & Kendall rank correlation + pergeseran peringkat).
5. Visualisasi pendukung (Top 10 bar comparison, yearly multi-line trends,
   dan scatter perbandingan publikasi vs author unik).
"""

import gc
import json
import time
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

try:
    from src.data_loader import build_venue_label_map, load_clean_data
except ImportError:
    from data_loader import build_venue_label_map, load_clean_data

OUTPUT_DIR = Path("output") / "task2"


def analyze_conference_popularity(
    df: pd.DataFrame,
    venue_display_map: Dict[str, str],
    output_dir: Path = OUTPUT_DIR,
    save_artifacts: bool = True,
) -> dict:
    """Menjalankan seluruh analisis Task 2 pada clean_inproceedings.parquet."""
    t0 = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    n_papers_total = len(df)

    # 1. Statistik per venue berbasis publikasi
    venue_paper_stats = (
        df.groupby("venue_key", as_index=False)
        .agg(
            n_papers=("paper_key", "count"),
            n_workshop_papers=("is_workshop", "sum"),
            in_conf_prefix=("in_conf_prefix", "first"),
            active_years=("year", "nunique"),
            first_year=("year", "min"),
            last_year=("year", "max"),
        )
    )

    # 2. Jumlah author unik per venue
    venue_codes_paper, unique_venues = pd.factorize(df["venue_key"], sort=True)
    authors_col = df["author_id"].tolist()
    n_auth_per_paper = np.fromiter((len(lst) for lst in authors_col), dtype=np.int32, count=len(authors_col))
    flat_authors = [a for lst in authors_col for a in lst]
    author_codes_flat, _ = pd.factorize(pd.Series(flat_authors, dtype="object"), sort=False)
    del flat_authors, authors_col

    venue_codes_flat = np.repeat(venue_codes_paper.astype(np.int64), n_auth_per_paper)
    # Total entri author-paper per venue
    v_entry_idx, v_entry_cnt = np.unique(venue_codes_flat, return_counts=True)
    total_entries_map = {str(unique_venues[i]): int(c) for i, c in zip(v_entry_idx, v_entry_cnt)}

    # Pasangan (author, venue) unik -> jumlah author unik per venue
    av_keys = (author_codes_flat.astype(np.int64) << 16) | venue_codes_flat
    av_unique = np.unique(av_keys)
    del av_keys, author_codes_flat, venue_codes_flat, venue_codes_paper, n_auth_per_paper
    gc.collect()

    v_uniq_idx, v_uniq_cnt = np.unique(av_unique & 0xFFFF, return_counts=True)
    unique_authors_map = {str(unique_venues[i]): int(c) for i, c in zip(v_uniq_idx, v_uniq_cnt)}
    del av_unique

    venue_stats = venue_paper_stats.copy()
    venue_stats["venue_name"] = venue_stats["venue_key"].map(venue_display_map)
    venue_stats["n_unique_authors"] = venue_stats["venue_key"].map(unique_authors_map).fillna(0).astype(int)
    venue_stats["n_author_entries"] = venue_stats["venue_key"].map(total_entries_map).fillna(0).astype(int)
    venue_stats["share_papers_pct"] = (venue_stats["n_papers"] / n_papers_total * 100.0).round(3)
    venue_stats["avg_authors_per_paper"] = (venue_stats["n_author_entries"] / venue_stats["n_papers"]).round(2)
    venue_stats["unique_authors_per_paper"] = (venue_stats["n_unique_authors"] / venue_stats["n_papers"]).round(2)
    venue_stats["workshop_pct"] = (venue_stats["n_workshop_papers"] / venue_stats["n_papers"] * 100.0).round(1)

    # Ranking berdasarkan jumlah publikasi & berdasarkan jumlah author unik
    venue_stats = venue_stats.sort_values(["n_papers", "n_unique_authors"], ascending=[False, False]).reset_index(drop=True)
    venue_stats["rank_papers"] = np.arange(1, len(venue_stats) + 1)

    venue_stats = venue_stats.sort_values(["n_unique_authors", "n_papers"], ascending=[False, False]).reset_index(drop=True)
    venue_stats["rank_authors"] = np.arange(1, len(venue_stats) + 1)
    venue_stats["rank_diff_papers_minus_authors"] = venue_stats["rank_papers"] - venue_stats["rank_authors"]

    # Urutkan kembali berdasarkan rank_papers sebagai urutan default
    venue_stats = venue_stats.sort_values("rank_papers").reset_index(drop=True)

    cols_view = [
        "rank_papers", "rank_authors", "rank_diff_papers_minus_authors",
        "venue_key", "venue_name", "n_papers", "share_papers_pct",
        "n_unique_authors", "avg_authors_per_paper", "unique_authors_per_paper", "workshop_pct",
    ]
    top10_papers = venue_stats.sort_values("rank_papers").head(10)[cols_view].reset_index(drop=True)
    top10_authors = venue_stats.sort_values("rank_authors").head(10)[cols_view].reset_index(drop=True)

    # Gabungan Top 10 Publikasi dan Top 10 Author Unik untuk tabel perbandingan
    top_union_keys = list(dict.fromkeys(top10_papers["venue_key"].tolist() + top10_authors["venue_key"].tolist()))
    rank_comparison = venue_stats[venue_stats["venue_key"].isin(top_union_keys)][cols_view].sort_values("rank_papers").reset_index(drop=True)

    # Korelasi ranking (semua 4.232 venue & Top 50 venue)
    rho_all, p_rho_all = spearmanr(venue_stats["n_papers"], venue_stats["n_unique_authors"])
    tau_all, p_tau_all = kendalltau(venue_stats["n_papers"], venue_stats["n_unique_authors"])

    top50 = venue_stats.head(50)
    rho_top50, p_rho_top50 = spearmanr(top50["n_papers"], top50["n_unique_authors"])
    tau_top50, p_tau_top50 = kendalltau(top50["n_papers"], top50["n_unique_authors"])

    top10_overlap = sorted(set(top10_papers["venue_key"]) & set(top10_authors["venue_key"]))
    only_in_top10_papers = sorted(set(top10_papers["venue_key"]) - set(top10_authors["venue_key"]))
    only_in_top10_authors = sorted(set(top10_authors["venue_key"]) - set(top10_papers["venue_key"]))

    corr_summary = {
        "n_venues_all": int(len(venue_stats)),
        "spearman_all": round(float(rho_all), 4),
        "spearman_all_p": float(p_rho_all),
        "kendall_all": round(float(tau_all), 4),
        "kendall_all_p": float(p_tau_all),
        "spearman_top50": round(float(rho_top50), 4),
        "spearman_top50_p": float(p_rho_top50),
        "kendall_top50": round(float(tau_top50), 4),
        "kendall_top50_p": float(p_tau_top50),
        "top10_overlap_count": len(top10_overlap),
        "only_in_top10_papers": [venue_display_map[k] for k in only_in_top10_papers],
        "only_in_top10_authors": [venue_display_map[k] for k in only_in_top10_authors],
    }

    # 3. Tren jumlah publikasi tahunan (2015-2025): total global & Top 10 konferensi
    yearly_global = (
        df.groupby("year", as_index=False)
        .agg(
            total_papers=("paper_key", "count"),
            unique_venues=("venue_key", "nunique"),
            workshop_papers=("is_workshop", "sum"),
        )
        .sort_values("year")
        .reset_index(drop=True)
    )
    yearly_global["yoy_growth_pct"] = (yearly_global["total_papers"].pct_change() * 100.0).round(2)

    top10_keys = top10_papers["venue_key"].tolist()
    yearly_top10 = (
        df[df["venue_key"].isin(top10_keys)]
        .groupby(["year", "venue_key"])["paper_key"]
        .count()
        .unstack(fill_value=0)
        .reindex(columns=top10_keys)
    )
    yearly_top10_named = yearly_top10.rename(columns=venue_display_map)

    # Ringkasan pertumbuhan 2015 -> 2024 (tahun penuh terakhir sebelum snapshot 2025 parsial)
    growth_rows = []
    for vk in top10_keys:
        vname = venue_display_map[vk]
        p2015 = int(yearly_top10.loc[2015, vk]) if 2015 in yearly_top10.index else 0
        p2019 = int(yearly_top10.loc[2019, vk]) if 2019 in yearly_top10.index else 0
        p2020 = int(yearly_top10.loc[2020, vk]) if 2020 in yearly_top10.index else 0
        p2024 = int(yearly_top10.loc[2024, vk]) if 2024 in yearly_top10.index else 0
        p2025 = int(yearly_top10.loc[2025, vk]) if 2025 in yearly_top10.index else 0
        pct_15_24 = round((p2024 - p2015) / p2015 * 100.0, 1) if p2015 > 0 else np.nan
        growth_rows.append({
            "venue_key": vk,
            "venue_name": vname,
            "papers_2015": p2015,
            "papers_2019": p2019,
            "papers_2020": p2020,
            "papers_2024": p2024,
            "papers_2025_partial": p2025,
            "growth_2015_to_2024_pct": pct_15_24,
            "peak_year": int(yearly_top10[vk].idxmax()),
            "peak_papers": int(yearly_top10[vk].max()),
        })
    growth_df = pd.DataFrame(growth_rows)

    print(f"\n[Task 2] Analisis popularitas selesai dalam {time.perf_counter() - t0:.2f} detik.")
    print("\n=== TOP 10 KONFERENSI BERDASARKAN JUMLAH PUBLIKASI ===")
    print(top10_papers[["rank_papers", "rank_authors", "venue_name", "n_papers", "n_unique_authors", "avg_authors_per_paper"]].to_string(index=False))
    print("\n=== TOP 10 KONFERENSI BERDASARKAN JUMLAH AUTHOR UNIK ===")
    print(top10_authors[["rank_authors", "rank_papers", "venue_name", "n_unique_authors", "n_papers", "avg_authors_per_paper"]].to_string(index=False))
    print("\n=== PERTUMBUHAN TOP 10 KONFERENSI (2015 -> 2024, + 2025 PARSIAL) ===")
    print(growth_df.to_string(index=False))
    print("\n=== KORELASI RANKING PUBLIKASI VS AUTHOR UNIK ===")
    print(json.dumps(corr_summary, indent=2))

    if save_artifacts:
        venue_stats[cols_view].to_csv(output_dir / "task2_all_venues_popularity.csv", index=False)
        top10_papers.to_csv(output_dir / "task2_top10_by_papers.csv", index=False)
        top10_authors.to_csv(output_dir / "task2_top10_by_authors.csv", index=False)
        rank_comparison.to_csv(output_dir / "task2_rank_comparison.csv", index=False)
        yearly_global.to_csv(output_dir / "task2_yearly_global.csv", index=False)
        yearly_top10_named.to_csv(output_dir / "task2_yearly_top10_conferences.csv")
        growth_df.to_csv(output_dir / "task2_top10_growth_2015_2024.csv", index=False)
        (output_dir / "task2_summary.json").write_text(
            json.dumps(corr_summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        plot_task2_top10_comparison(top10_papers, top10_authors, output_dir / "task2_top10_papers_vs_authors.png")
        plot_task2_yearly_trends(yearly_global, yearly_top10_named, output_dir / "task2_yearly_trends_2015_2025.png")
        plot_task2_rank_scatter(venue_stats, output_dir / "task2_rank_comparison_scatter.png")

    return {
        "venue_stats": venue_stats,
        "top10_papers": top10_papers,
        "top10_authors": top10_authors,
        "rank_comparison": rank_comparison,
        "yearly_global": yearly_global,
        "yearly_top10": yearly_top10_named,
        "growth_df": growth_df,
        "corr_summary": corr_summary,
    }


def plot_task2_top10_comparison(top10_papers: pd.DataFrame, top10_authors: pd.DataFrame, out_path: Path):
    """Side-by-side horizontal bar chart: Top 10 Publikasi vs Top 10 Author Unik."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Panel kiri: Top 10 Jumlah Publikasi
    y1 = np.arange(len(top10_papers))
    b1 = axes[0].barh(y1, top10_papers["n_papers"], color="#2B6CB0", edgecolor="black", alpha=0.88)
    axes[0].set_yticks(y1)
    axes[0].set_yticklabels(top10_papers["venue_name"], fontsize=9.5, fontweight="bold")
    axes[0].invert_yaxis()
    axes[0].set_title("Top 10 Konferensi Berdasarkan Jumlah Publikasi\n(DBLP 2015–2025)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Jumlah Publikasi (paper)")
    axes[0].set_xlim(0, float(top10_papers["n_papers"].max()) * 1.24)
    axes[0].grid(axis="x", linestyle="--", alpha=0.55)
    for bar, (_, r) in zip(b1, top10_papers.iterrows()):
        w = bar.get_width()
        axes[0].text(w + 350, bar.get_y() + bar.get_height() / 2, f"{int(w):,} (#{int(r['rank_authors'])} auth)", va="center", fontsize=8, fontweight="bold", color="#1A365D")

    # Panel kanan: Top 10 Jumlah Author Unik
    y2 = np.arange(len(top10_authors))
    b2 = axes[1].barh(y2, top10_authors["n_unique_authors"], color="#2F855A", edgecolor="black", alpha=0.88)
    axes[1].set_yticks(y2)
    axes[1].set_yticklabels(top10_authors["venue_name"], fontsize=9.5, fontweight="bold")
    axes[1].invert_yaxis()
    axes[1].set_title("Top 10 Konferensi Berdasarkan Jumlah Author Unik\n(DBLP 2015–2025)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Jumlah Author Unik")
    axes[1].set_xlim(0, float(top10_authors["n_unique_authors"].max()) * 1.26)
    axes[1].grid(axis="x", linestyle="--", alpha=0.55)
    for bar, (_, r) in zip(b2, top10_authors.iterrows()):
        w = bar.get_width()
        axes[1].text(w + 700, bar.get_y() + bar.get_height() / 2, f"{int(w):,} (#{int(r['rank_papers'])} pub)", va="center", fontsize=8, fontweight="bold", color="#1A365D")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[Visualisasi] Disimpan ke: {out_path}")


def plot_task2_yearly_trends(yearly_global: pd.DataFrame, yearly_top10: pd.DataFrame, out_path: Path):
    """2-panel chart: (kiri) tren publikasi global 2015-2025, (kanan) tren Top 10 konferensi."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.8))

    # Panel kiri: Total publikasi DBLP per tahun
    yrs = yearly_global["year"].to_numpy()
    paps = yearly_global["total_papers"].to_numpy()
    colors_bar = ["#2B6CB0" if y < 2025 else "#A0AEC0" for y in yrs]
    bars = axes[0].bar(yrs, paps / 1000.0, color=colors_bar, edgecolor="black", alpha=0.88)
    axes[0].set_title("Total Aktivitas Publikasi Konferensi/Workshop per Tahun\n(2025 abu-abu = parsial, snapshot Des 2025)", fontsize=10.5, fontweight="bold")
    axes[0].set_xlabel("Tahun Publikasi")
    axes[0].set_ylabel("Jumlah Publikasi (Ribu)")
    axes[0].set_xticks(yrs)
    axes[0].set_xticklabels(yrs, rotation=45)
    axes[0].set_ylim(0, float(paps.max() / 1000.0) * 1.15)
    axes[0].grid(axis="y", linestyle="--", alpha=0.55)
    for bar, val in zip(bars, paps):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{val/1000:.1f}k", ha="center", va="bottom", fontsize=7.8, fontweight="bold")

    # Panel kanan: Tren Top 10 konferensi
    cmap = plt.get_cmap("tab10")
    for idx, col in enumerate(yearly_top10.columns):
        axes[1].plot(yearly_top10.index, yearly_top10[col], marker="o", linewidth=2.1, markersize=5, color=cmap(idx), label=col)
    axes[1].axvspan(2024.5, 2025.3, color="#EDF2F7", alpha=0.8, label="2025 (parsial)")
    axes[1].set_title("Tren Jumlah Publikasi Tahunan Top 10 Konferensi (2015–2025)", fontsize=10.5, fontweight="bold")
    axes[1].set_xlabel("Tahun Publikasi")
    axes[1].set_ylabel("Jumlah Publikasi per Tahun")
    axes[1].set_xticks(yearly_top10.index)
    axes[1].set_xticklabels(yearly_top10.index, rotation=45)
    axes[1].grid(True, linestyle="--", alpha=0.55)
    axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[Visualisasi] Disimpan ke: {out_path}")


def plot_task2_rank_scatter(venue_stats: pd.DataFrame, out_path: Path):
    """Scatter plot Jumlah Publikasi vs Jumlah Author Unik (Top 100 + anotasi Top 15)."""
    top100 = venue_stats.head(100)
    fig, ax = plt.subplots(figsize=(10.5, 6))
    sc = ax.scatter(
        top100["n_papers"],
        top100["n_unique_authors"],
        c=top100["avg_authors_per_paper"],
        cmap="plasma",
        s=55,
        alpha=0.82,
        edgecolors="black",
        linewidths=0.5,
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Rata-rata Author per Paper")

    # Anotasi venue yang masuk Top 10 publikasi atau Top 10 author unik
    highlight = venue_stats[(venue_stats["rank_papers"] <= 10) | (venue_stats["rank_authors"] <= 10)]
    for _, r in highlight.iterrows():
        ax.annotate(
            r["venue_name"],
            (r["n_papers"], r["n_unique_authors"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8.2,
            fontweight="bold",
            color="#1A202C",
        )

    ax.set_title("Perbandingan Jumlah Publikasi vs Jumlah Author Unik (Top 100 Konferensi)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Jumlah Publikasi (2015–2025)")
    ax.set_ylabel("Jumlah Author Unik (2015–2025)")
    ax.grid(True, linestyle="--", alpha=0.55)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[Visualisasi] Disimpan ke: {out_path}")


if __name__ == "__main__":
    df_clean, _ = load_clean_data(verify_hash=True)
    _, v_display_map = build_venue_label_map(df_clean)
    analyze_conference_popularity(df_clean, v_display_map, OUTPUT_DIR, save_artifacts=True)
