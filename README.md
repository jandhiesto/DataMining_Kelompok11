# Data Mining Kelompok 11
**Conference Popularity and Association Pattern Mining Using DBLP Dataset**

Proyek ini menganalisis popularitas konferensi ilmiah Computer Science dan pola hubungan asosiasi antar-konferensi berdasarkan riwayat publikasi para peneliti pada dataset DBLP (snapshot `dblp-2025-12-01`, periode 2015–2025, tipe `inproceedings`).

---

## Notebook Utama & Struktur Branch

Sesuai kesepakatan commit terbaru `Role1`, notebook utama proyek adalah **`Notebook-dblp_2025_12_01.ipynb`**.

- **`Role1` (Part A & Part B — Task 1):**
  - Ekstraksi streaming XML DBLP (`1.901.774` record mentah 2015–2025).
  - Pembersihan data lengkap (`T00`–`T06`): missing values, konsistensi tahun, normalisasi `author_id` (sufiks homonim dipertahankan), normalisasi `venue_key` & `venue_name` (1-ke-1), deduplikasi Tier-1 & Tier-2.
  - Ekspor artefak akhir: `data/clean_inproceedings.parquet` (`1.899.038` paper bersih, `1.738.301` author unik, `4.232` venue unik) beserta `data/clean_summary.json`.
- **`Role2` (Part C — Task 3, Task 4, Task 5, Task 6):**
  - **Task 3 — Transactional Database Construction:** Membangun *Author-Based* dan *Author-Year-Based* Transactional Databases menggunakan `venue_key` kanonik (vectorized NumPy + CSR sparse matrix), 10 contoh transaksi masing-masing, statistik distribusi panjang item, serta analisis perbedaan/kelebihan/keterbatasan.
  - **Task 4 — Frequent Pattern Mining:** Implementasi Apriori dan FP-Growth pada *Author-Based Transactional Database* dengan 4 nilai *minimum support* (`0.1%`, `0.5%`, `1%`, `2%`), identifikasi 1-/2-/3-itemsets, Top 10 itemsets, dan interpretasi pola.
  - **Task 5 — Association Rule Mining:** Pembangkitan rules (`min_confidence = 50%`), penghitungan Support, Confidence, Lift, Top 10 Lift, serta uji korelasi & analisis *Support vs Lift*.
  - **Task 6 — Algorithm Performance Comparison:** Komparasi Apriori vs FP-Growth (jumlah itemsets, jumlah rules, runtime, peak RAM, dan verifikasi kesamaan eksak 100%).

---

## Persiapan Data & Cara Menjalankan Role 2

1. **Unduh Dataset Bersih (dari Google Drive tim -> folder `data_baru/`):**
   - Unduh **`clean_inproceedings.parquet`** (`205.006.040` bytes, SHA-256: `778066063eb72e3db573fc8a50669f3f1e9b2faa6903393d47812242227fcf94`).
   - Simpan ke: `data/clean_inproceedings.parquet`.
   - *(Berkas `.json` seperti `clean_summary.json`, `dblp_profile.json`, dan `raw_counts.json` sudah otomatis tersedia di repositori Git).*

2. **Pasang Dependensi Python:**
   ```bash
   pip install pandas numpy scipy matplotlib pyarrow mlxtend
   ```

3. **Menjalankan via Jupyter Notebook (Part C — Standalone):**
   Buka `Notebook-dblp_2025_12_01.ipynb` dan langsung jalankan sel mulai dari **`R2-0`** (Part C) hingga sel terakhir (`T6-1`). Tidak perlu menyimpan berkas XML mentah 4,9 GB maupun menjalankan ulang Part A/B.

4. **Atau Menjalankan via Skrip Modular (CLI):**
   ```bash
   python src/role2_mining.py
   ```
   Seluruh tabel CSV, ringkasan JSON, dan grafik PNG Role 2 akan diekspor ke folder `output/role2/`.
