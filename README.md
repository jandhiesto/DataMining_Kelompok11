# Data Mining Kelompok 11
**Conference Popularity and Association Pattern Mining Using DBLP Dataset**

Proyek ini menganalisis popularitas konferensi ilmiah Computer Science dan pola hubungan asosiasi antar-konferensi berdasarkan riwayat publikasi para peneliti pada dataset DBLP (periode 2015–2025, tipe `inproceedings`).

---

## Struktur Pengerjaan Branch

- **`Role1`**:
  - Dikerjakan oleh Anggota 1.
  - Penyiapan lingkungan, ekstraksi streaming data mentah XML DBLP (2015–2025), pemahaman struktur data bibliografi, dan tahapan awal data preprocessing.
- **`Role2` (Branch ini)**:
  - Dikerjakan oleh Anggota 2.
  - **Task 3: Transactional Database Construction**
    - Membangun *Author-Based* dan *Author-Year-Based* Transactional Databases.
    - Menghitung statistik distribusi panjang item dan analisis perbandingan kedua representasi.
  - **Task 4: Frequent Pattern Mining**
    - Implementasi algoritma Apriori dan FP-Growth pada Author-Based Transaction Database.
    - Eksperimen pada 4 level minimum support: `0.1%`, `0.5%`, `1.0%`, dan `2.0%`.
    - Ekstraksi k-itemsets (1-itemset, 2-itemset, 3-itemset) dan 10 frequent itemsets teratas.
  - **Task 5: Association Rule Mining**
    - Pembangkitan aturan asosiasi dengan *minimum confidence* 50%.
    - Penghitungan metrik Support, Confidence, dan Lift.
    - Identifikasi 10 aturan asosiasi dengan Lift tertinggi dan analisis hubungan support vs lift.
  - **Task 6: Algorithm Performance Comparison**
    - Pengukuran komparasi Apriori vs FP-Growth pada parameter yang sama (runtime, penggunaan memori via `tracemalloc`, jumlah itemset & rules).
    - Menjawab 3 pertanyaan evaluasi metodologi data mining.

---

## Cara Menjalankan

### Persyaratan Lingkungan (Dependencies)
Pastikan pustaka Python berikut telah terpasang:
```bash
pip install pandas numpy matplotlib pyarrow mlxtend
```

### 1. Menjalankan via Jupyter Notebook
Buka dan jalankan seluruh sel pada berkas utama:
```bash
jupyter notebook ProjectDataMining_dblpDec2025.ipynb
```

### 2. Menjalankan via Script Python Modular
Anda juga dapat menjalankan pipeline komputasi Task 3–6 langsung melalui terminal:
```bash
python src/role2_mining.py
```
Seluruh gambar grafik visualisasi dan tabel hasil ekspor akan otomatis disimpan di folder `output/`.

---

## Catatan Dataset & Penanganan File Parquet
- File data mentah `.parquet` dikecualikan dari repositori Git melalui `.gitignore` karena ukurannya yang besar.
- Pipeline ini dilengkapi mekanisme **Data Fallback Cerdas**:
  1. Jika `data/clean_inproceedings.parquet` atau `data/raw_inproceedings.parquet` ada (hasil copy dari laptop Role 1), pipeline otomatis memproses seluruh dataset penuh.
  2. Jika file parquet belum disalin ke komputer lokal, pipeline otomatis menggunakan `data/clean_inproceedings_sample.parquet` sehingga seluruh kode, plot grafik, dan tabel tetap dapat dieksekusi tanpa error.
