# Data Mining Kelompok 11
**Conference Popularity and Association Pattern Mining Using DBLP Dataset**

Proyek ini menganalisis popularitas konferensi ilmiah Computer Science serta pola asosiasi antar-konferensi (statis dan temporal) berdasarkan riwayat publikasi para peneliti pada dataset DBLP (snapshot `dblp-2025-12-01`, periode **2015–2025**, tipe `inproceedings`).

---

## Notebook Utama & Cakupan Tugas

Notebook utama terintegrasi yang memuat **seluruh tugas secara utuh** (Part A s.d. Part E) adalah **`Notebook-dblp_2025_12_01.ipynb`**:

- **Part A & Part B — Task 1 (Data Understanding & Preprocessing — Role 1):**
  - Ekstraksi streaming XML DBLP (`1.901.774` record mentah 2015–2025).
  - Pembersihan lengkap (`T00`–`T06`): missing values, konsistensi tahun, normalisasi `author_id` (sufiks homonim dipertahankan), normalisasi `venue_key` & `venue_name` (1-ke-1), deduplikasi Tier-1 & Tier-2.
  - Artefak bersih: `data/clean_inproceedings.parquet` (**1.899.038** publikasi bersih, **1.738.301** author unik, **4.232** venue unik) + `data/clean_summary.json`.
- **Part B2 — Task 2 (Conference Popularity Analysis):**
  - Skrip modular: `src/task2_popularity.py` | Output: `output/task2/`.
  - Top 10 konferensi berdasarkan jumlah publikasi (`CVPR` #1: `22.741`) & author unik (`CVPR` #1: `47.421`), tren tahunan 2015–2025, korelasi ranking (Spearman/Kendall), dan 3 visualisasi.
- **Part C — Task 3, Task 4, Task 5, Task 6 (Core Pattern Mining & Benchmarking — Role 2):**
  - Skrip modular: `src/role2_mining.py` | Output: `output/role2/`.
  - **Task 3:** Konstruksi *Author-Based* (`1.738.301` transaksi) dan *Author-Year-Based* (`3.895.225` transaksi), 10 contoh masing-masing, distribusi item, dan analisis komparasi.
  - **Task 4:** Apriori & FP-Growth pada 4 *minimum support* (`0.1%`, `0.5%`, `1.0%`, `2.0%`), 1-/2-/3-itemsets, Top 10 itemsets, dan interpretasi kluster domain.
  - **Task 5:** Association Rules (`min_confidence = 50%`, `865` rules di `0.1%`), Top 10 Lift (maks `148.92` multi-item; `81.81` untuk `1->1` `{ECOC}->{OFC}`), dan uji korelasi *Support vs Lift*.
  - **Task 6:** Komparasi performa Apriori (`low_memory=True`) vs FP-Growth (itemsets & rules 100% identik, runtime, peak RAM, dan evaluasi empiris).
- **Part D — Task E (Temporal Association Analysis — Role 3):**
  - Skrip modular: `src/role3_temporal.py` | Notebook khusus: `notebooks/task3b_taskE_temporal.ipynb` | Output: `output/role3/`.
  - Analisis lintas 4 periode (`2015–2017`, `2018–2020`, `2021–2023`, `2024–2025`) pada *Author-Year-Based* (`min_support = 0.05%`, `min_confidence = 25%` + uji sensitivitas `25%–50%`), pola bertahan (`3` di 4/4 periode, `15` di $\ge 3/4$ periode), tren pergeseran (`emerging`/`fading`), overlap Jaccard, dan visualisasi evolusi Lift & support venue.
- **Part E — Bagian H & Keterbatasan (Ketentuan F.8):**
  - Jawaban empiris lengkap atas **7 Pertanyaan Kesimpulan** serta dokumentasi keterbatasan dataset dan asumsi analisis.

---

## Persiapan Data & Cara Menjalankan

1. **Unduh Dataset Bersih (dari Google Drive tim -> folder `data_baru/`):**
   - Unduh **`clean_inproceedings.parquet`** (`205.006.040` bytes, SHA-256: `778066063eb72e3db573fc8a50669f3f1e9b2faa6903393d47812242227fcf94`).
   - Simpan ke: `data/clean_inproceedings.parquet`.

2. **Pasang Dependensi Python:**
   ```bash
   pip install pandas numpy scipy matplotlib pyarrow mlxtend
   ```

3. **Menjalankan via Skrip Modular (CLI):**
   ```bash
   python src/task2_popularity.py       # Task 2 -> output/task2/
   python src/role2_mining.py           # Task 3, 4, 5, 6 -> output/role2/
   python src/role3_temporal.py --verify  # Task 3B & Task E -> output/role3/
   ```

4. **Menjalankan via Jupyter Notebook:**
   - Master notebook lengkap: `Notebook-dblp_2025_12_01.ipynb` (Part B2, C, D dapat dijalankan *standalone* langsung dari `data/clean_inproceedings.parquet` tanpa perlu XML mentah 4,9 GB).
   - Notebook khusus Role 3: `notebooks/task3b_taskE_temporal.ipynb`.
