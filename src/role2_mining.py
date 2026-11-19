import time
import tracemalloc
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
from data_loader import load_clean_data

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


# ==============================================================================
# TASK 3: TRANSACTIONAL DATABASE CONSTRUCTION
# ==============================================================================
def construct_transactional_databases(df):
    """
    Membangun dua jenis representasi transaksi dari dataset DBLP:
    A. Author-Based: Satu transaksi = 1 author, item = daftar venue unik (2015-2025).
    B. Author-Year-Based: Satu transaksi = kombinasi (author, year), item = daftar venue unik pada tahun tersebut.
    """
    print("\n" + "="*80)
    print("TASK 3: TRANSACTIONAL DATABASE CONSTRUCTION")
    print("="*80)

    # Explode authors sehingga setiap baris berisi (paper_key, author, year, venue)
    df_exploded = df.explode("authors").rename(columns={"authors": "author"}).copy()
    df_exploded["author"] = df_exploded["author"].astype(str).str.strip()
    df_exploded = df_exploded[df_exploded["author"] != ""].copy()

    # A. Author-Based Transaction Database
    # Groupby author -> set of unique venues
    author_grouped = (
        df_exploded.groupby("author")["venue"]
        .apply(lambda s: sorted(list(set(s))))
        .reset_index(name="venues")
    )
    author_tx = author_grouped["venues"].tolist()

    # B. Author-Year-Based Transaction Database
    # Groupby (author, year) -> set of unique venues
    author_year_grouped = (
        df_exploded.groupby(["author", "year"])["venue"]
        .apply(lambda s: sorted(list(set(s))))
        .reset_index(name="venues")
    )
    author_year_tx = author_year_grouped["venues"].tolist()

    # Statistik Dasar
    author_tx_lens = [len(tx) for tx in author_tx]
    author_year_tx_lens = [len(tx) for tx in author_year_tx]

    unique_venues_a = len(set(v for tx in author_tx for v in tx))
    unique_venues_ay = len(set(v for tx in author_year_tx for v in tx))

    stats_df = pd.DataFrame({
        "Metrik": [
            "Total Transaksi",
            "Jumlah Item Unik (Venues)",
            "Panjang Min Item",
            "Panjang Max Item",
            "Rata-rata Item per Transaksi",
            "Median Panjang Item",
            "Std Dev Panjang Item"
        ],
        "Author-Based": [
            f"{len(author_tx):,}",
            unique_venues_a,
            min(author_tx_lens),
            max(author_tx_lens),
            f"{np.mean(author_tx_lens):.2f}",
            np.median(author_tx_lens),
            f"{np.std(author_tx_lens):.2f}"
        ],
        "Author-Year-Based": [
            f"{len(author_year_tx):,}",
            unique_venues_ay,
            min(author_year_tx_lens),
            max(author_year_tx_lens),
            f"{np.mean(author_year_tx_lens):.2f}",
            np.median(author_year_tx_lens),
            f"{np.std(author_year_tx_lens):.2f}"
        ]
    })

    print("\n--- STATISTIK KOMPARASI REPRESENTASI TRANSAKSI ---")
    print(stats_df.to_string(index=False))

    print("\n--- 10 CONTOH TRANSAKSI AUTHOR-BASED ---")
    for i in range(min(10, len(author_grouped))):
        print(f"[{i+1}] {author_grouped.iloc[i]['author']}: {author_grouped.iloc[i]['venues']}")

    print("\n--- 10 CONTOH TRANSAKSI AUTHOR-YEAR-BASED ---")
    for i in range(min(10, len(author_year_grouped))):
        row = author_year_grouped.iloc[i]
        print(f"[{i+1}] {row['author']} ({row['year']}): {row['venues']}")

    # Plot Distribusi Panjang Item
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(author_tx_lens, bins=range(1, max(author_tx_lens)+2), color="#2B6CB0", edgecolor="black", alpha=0.85)
    plt.title("Distribusi Jumlah Item: Author-Based", fontsize=11, fontweight="bold")
    plt.xlabel("Jumlah Item (Konferensi) per Transaksi")
    plt.ylabel("Frekuensi")
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.hist(author_year_tx_lens, bins=range(1, max(author_year_tx_lens)+2), color="#C05621", edgecolor="black", alpha=0.85)
    plt.title("Distribusi Jumlah Item: Author-Year-Based", fontsize=11, fontweight="bold")
    plt.xlabel("Jumlah Item (Konferensi) per Transaksi")
    plt.ylabel("Frekuensi")
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    dist_plot_path = OUTPUT_DIR / "task3_transaction_distribution.png"
    plt.savefig(dist_plot_path, dpi=300)
    plt.close()
    print(f"\n[Visualisasi] Plot distribusi disimpan ke: {dist_plot_path}")

    return author_tx, author_year_tx, stats_df


# ==============================================================================
# TASK 4 & TASK 6: FREQUENT PATTERN MINING & ALGORITHM BENCHMARKING
# ==============================================================================
def run_frequent_pattern_mining(author_tx, min_supports=[0.001, 0.005, 0.01, 0.02]):
    """
    Menjalankan Apriori dan FP-Growth pada Author-Based Transaction Database
    dengan 4 nilai minimum support: 0.1%, 0.5%, 1%, dan 2%.
    Mengukur waktu eksekusi (runtime) dan memori puncak (peak memory).
    """
    print("\n" + "="*80)
    print("TASK 4 & TASK 6: FREQUENT PATTERN MINING & PERFORMANCE COMPARISON")
    print("="*80)

    # Transformasi transaksi ke format Sparse Matrix untuk efisiensi RAM masif
    print("[Encoder] Melakukan One-Hot Encoding dengan matriks sparse...")
    te = TransactionEncoder()
    te_ary = te.fit(author_tx).transform(author_tx, sparse=True)
    df_sparse = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)
    print(f"[Encoder] Ukuran matriks transaksi: {df_sparse.shape[0]:,} transaksi x {df_sparse.shape[1]} venue unik.")

    benchmark_records = []
    fpm_results = {}

    for s in min_supports:
        sup_pct = f"{s*100:.1f}%"
        print(f"\n>>> Running Experiments pada Minimum Support: {sup_pct} ({s}) <<<")

        # A. FP-GROWTH
        tracemalloc.start()
        t0 = time.perf_counter()
        frequent_fp = fpgrowth(df_sparse, min_support=s, use_colnames=True)
        t_fp = time.perf_counter() - t0
        _, peak_mem_fp = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        mem_fp_mb = peak_mem_fp / (1024 * 1024)

        # Hitung panjang k-itemsets
        if not frequent_fp.empty:
            frequent_fp["length"] = frequent_fp["itemsets"].apply(len)
        else:
            frequent_fp["length"] = 0

        # B. APRIORI
        tracemalloc.start()
        t0 = time.perf_counter()
        frequent_ap = apriori(df_sparse, min_support=s, use_colnames=True)
        t_ap = time.perf_counter() - t0
        _, peak_mem_ap = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        mem_ap_mb = peak_mem_ap / (1024 * 1024)

        if not frequent_ap.empty:
            frequent_ap["length"] = frequent_ap["itemsets"].apply(len)
        else:
            frequent_ap["length"] = 0

        # Verifikasi kesamaan himpunan itemset
        set_fp = set(frequent_fp["itemsets"].apply(frozenset))
        set_ap = set(frequent_ap["itemsets"].apply(frozenset))
        is_identical = (set_fp == set_ap)

        # Hitung association rules (min_conf = 0.50) untuk Task 6
        if len(frequent_fp) > 0:
            rules_fp = association_rules(frequent_fp, metric="confidence", min_threshold=0.50)
            rules_ap = association_rules(frequent_ap, metric="confidence", min_threshold=0.50)
            num_rules_fp = len(rules_fp)
            num_rules_ap = len(rules_ap)
        else:
            num_rules_fp = 0
            num_rules_ap = 0

        # Hitung 1-itemsets, 2-itemsets, 3-itemsets
        c1 = int((frequent_fp["length"] == 1).sum()) if not frequent_fp.empty else 0
        c2 = int((frequent_fp["length"] == 2).sum()) if not frequent_fp.empty else 0
        c3 = int((frequent_fp["length"] == 3).sum()) if not frequent_fp.empty else 0
        c_more = int((frequent_fp["length"] >= 4).sum()) if not frequent_fp.empty else 0

        benchmark_records.append({
            "Min Support": sup_pct,
            "Min Support (Float)": s,
            "Total Itemsets (FP)": len(frequent_fp),
            "Total Itemsets (Apriori)": len(frequent_ap),
            "Itemsets Identik": is_identical,
            "1-itemsets": c1,
            "2-itemsets": c2,
            "3-itemsets": c3,
            ">=4-itemsets": c_more,
            "Total Rules (Conf 50%)": num_rules_fp,
            "Runtime FP-Growth (s)": round(t_fp, 4),
            "Runtime Apriori (s)": round(t_ap, 4),
            "Speedup (Apriori/FP)": round(t_ap / t_fp, 2) if t_fp > 0 else 1.0,
            "Peak RAM FP-Growth (MB)": round(mem_fp_mb, 2),
            "Peak RAM Apriori (MB)": round(mem_ap_mb, 2),
        })

        # Simpan hasil untuk analisis lebih lanjut
        fpm_results[s] = {
            "fp_itemsets": frequent_fp.sort_values(by="support", ascending=False).reset_index(drop=True),
            "ap_itemsets": frequent_ap.sort_values(by="support", ascending=False).reset_index(drop=True),
            "rules": rules_fp.sort_values(by="lift", ascending=False).reset_index(drop=True) if len(frequent_fp) > 0 else pd.DataFrame()
        }

        # Tampilkan 10 Top Itemsets untuk min support ini
        print(f"\n[Top 10 Frequent Itemsets - Support {sup_pct}]")
        top10 = fpm_results[s]["fp_itemsets"].head(10)[["support", "itemsets", "length"]].copy()
        top10["itemsets"] = top10["itemsets"].apply(lambda x: ", ".join(list(x)))
        print(top10.to_string(index=False))

    benchmark_df = pd.DataFrame(benchmark_records)
    print("\n" + "="*80)
    print("TABEL RINGKASAN BENCHMARK PERFORMA (TASK 4 & TASK 6)")
    print("="*80)
    cols_display = [
        "Min Support", "Total Itemsets (FP)", "1-itemsets", "2-itemsets", "3-itemsets",
        "Total Rules (Conf 50%)", "Runtime FP-Growth (s)", "Runtime Apriori (s)", "Speedup (Apriori/FP)",
        "Peak RAM FP-Growth (MB)", "Peak RAM Apriori (MB)", "Itemsets Identik"
    ]
    print(benchmark_df[cols_display].to_string(index=False))

    plot_performance_comparison(benchmark_df)

    return fpm_results, benchmark_df


def plot_performance_comparison(benchmark_df):
    """Menghasilkan grafik visual komparasi performa Apriori vs FP-Growth."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    x_labels = benchmark_df["Min Support"].tolist()
    x = np.arange(len(x_labels))
    width = 0.35

    # Runtime Comparison
    axes[0].bar(x - width/2, benchmark_df["Runtime Apriori (s)"], width, label="Apriori", color="#E53E3E")
    axes[0].bar(x + width/2, benchmark_df["Runtime FP-Growth (s)"], width, label="FP-Growth", color="#3182CE")
    axes[0].set_title("Runtime Comparison (Detik)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Minimum Support")
    axes[0].set_ylabel("Waktu Eksekusi (Detik)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(x_labels)
    axes[0].legend()
    axes[0].grid(axis="y", linestyle="--", alpha=0.6)

    # Peak Memory Comparison
    axes[1].bar(x - width/2, benchmark_df["Peak RAM Apriori (MB)"], width, label="Apriori", color="#DD6B20")
    axes[1].bar(x + width/2, benchmark_df["Peak RAM FP-Growth (MB)"], width, label="FP-Growth", color="#38A169")
    axes[1].set_title("Peak Memory Usage (MB)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Minimum Support")
    axes[1].set_ylabel("Penggunaan Memori (MB)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(x_labels)
    axes[1].legend()
    axes[1].grid(axis="y", linestyle="--", alpha=0.6)

    # Itemsets & Rules Count
    axes[2].plot(x_labels, benchmark_df["Total Itemsets (FP)"], marker="o", linewidth=2.5, color="#805AD5", label="Frequent Itemsets")
    axes[2].plot(x_labels, benchmark_df["Total Rules (Conf 50%)"], marker="s", linewidth=2.5, color="#D69E2E", label="Rules (Conf >= 50%)")
    axes[2].set_title("Jumlah Itemsets & Rules vs Min Support", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Minimum Support")
    axes[2].set_ylabel("Jumlah Himpunan")
    axes[2].legend()
    axes[2].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    chart_path = OUTPUT_DIR / "task6_algorithm_performance_comparison.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"\n[Visualisasi] Grafik perbandingan performa disimpan ke: {chart_path}")


# ==============================================================================
# TASK 5: ASSOCIATION RULE MINING
# ==============================================================================
def run_association_rule_mining(fpm_results, target_support=0.001):
    """
    Membangkitkan dan menganalisis Association Rules dari Frequent Itemsets
    menggunakan minimum confidence 50%.
    Menghitung Support, Confidence, dan Lift serta mengekstrak 10 rules dengan Lift tertinggi.
    """
    print("\n" + "="*80)
    print("TASK 5: ASSOCIATION RULE MINING (MIN CONFIDENCE 50%)")
    print("="*80)

    if target_support in fpm_results and not fpm_results[target_support]["fp_itemsets"].empty:
        fi = fpm_results[target_support]["fp_itemsets"]
        used_sup = target_support
    else:
        used_sup = min(fpm_results.keys())
        fi = fpm_results[used_sup]["fp_itemsets"]

    print(f"[Rules] Membangkitkan aturan asosiasi dari {len(fi)} frequent itemsets (Min Support: {used_sup*100:.1f}%)...")
    rules = association_rules(fi, metric="confidence", min_threshold=0.50)

    if rules.empty:
        print("[Rules] Tidak ada aturan asosiasi yang memenuhi threshold confidence 50%.")
        return pd.DataFrame()

    rules["antecedents_str"] = rules["antecedents"].apply(lambda s: "{" + ", ".join(list(s)) + "}")
    rules["consequents_str"] = rules["consequents"].apply(lambda s: "{" + ", ".join(list(s)) + "}")
    rules["rule_str"] = rules["antecedents_str"] + " -> " + rules["consequents_str"]

    rules_sorted_lift = rules.sort_values(by="lift", ascending=False).reset_index(drop=True)
    top10_lift = rules_sorted_lift.head(10).copy()

    cols_show = ["rule_str", "support", "confidence", "lift"]
    print("\n--- TOP 10 ASSOCIATION RULES BERDASARKAN LIFT TERTINGGI ---")
    print(top10_lift[cols_show].to_string(index=False))

    print("\n--- ANALISIS KORELASI SUPPORT VS LIFT ---")
    corr = rules["support"].corr(rules["lift"])
    print(f"Koefisien Korelasi Pearson antara Support dan Lift: {corr:.4f}")
    if corr < 0:
        print("KESIMPULAN: Korelasi negatif! Aturan dengan Lift tertinggi cenderung memiliki support lebih rendah/sedang,")
        print("membuktikan bahwa konferensi spesialis/eksklusif menghasilkan Lift jauh lebih tinggi dibanding konferensi umum.")

    # Visualisasi Bar Chart Top 10 Rules Lift
    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(top10_lift))
    plt.barh(y_pos, top10_lift["lift"], color="#276749", edgecolor="black", alpha=0.85)
    plt.yticks(y_pos, top10_lift["rule_str"], fontsize=9)
    plt.gca().invert_yaxis()  # Lift tertinggi di atas
    plt.title("Top 10 Association Rules Berdasarkan Nilai Lift (Min Conf 50%)", fontsize=11, fontweight="bold")
    plt.xlabel("Nilai Lift")
    plt.grid(axis="x", linestyle="--", alpha=0.6)

    for i, v in enumerate(top10_lift["lift"]):
        plt.text(v + 0.1, i, f"{v:.2f}", va="center", fontsize=8.5, fontweight="bold", color="#1A365D")

    plt.tight_layout()
    rules_plot_path = OUTPUT_DIR / "task5_top10_rules_lift.png"
    plt.savefig(rules_plot_path, dpi=300)
    plt.close()
    print(f"\n[Visualisasi] Plot Top 10 Rules Lift disimpan ke: {rules_plot_path}")

    # Simpan hasil rules ke CSV
    rules_csv_path = OUTPUT_DIR / "association_rules_top_lift.csv"
    rules_sorted_lift[cols_show].to_csv(rules_csv_path, index=False)
    print(f"[Export] Seluruh rules disimpan ke: {rules_csv_path}")

    return rules_sorted_lift


# ==============================================================================
# PIPELINE UTAMA ROLE 2
# ==============================================================================
def execute_role2_pipeline():
    """Menjalankan seluruh alur kerja Role 2 secara otomatis."""
    print("="*80)
    print("PIPELINE EKSEKUSI TUGAS DATA MINING: ROLE 2")
    print("DataMining_Kelompok11 - Branch Role2")
    print("="*80)

    # 1. Load Data
    df = load_clean_data()

    # 2. Task 3: Transactional Database Construction
    author_tx, author_year_tx, stats_df = construct_transactional_databases(df)

    # 3. Task 4 & Task 6: Frequent Pattern Mining & Benchmarking
    fpm_results, benchmark_df = run_frequent_pattern_mining(
        author_tx,
        min_supports=[0.001, 0.005, 0.01, 0.02]
    )

    # 4. Task 5: Association Rule Mining
    rules_df = run_association_rule_mining(fpm_results, target_support=0.001)

    print("\n" + "="*80)
    print("SEMUA TUGAS ROLE 2 BERHASIL DISELESAIKAN!")
    print(f"Seluruh gambar visualisasi dan tabel hasil tersimpan di folder: {OUTPUT_DIR.resolve()}")
    print("="*80)


if __name__ == "__main__":
    execute_role2_pipeline()
