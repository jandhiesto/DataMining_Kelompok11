"""
Module data_loader.py
Fungsi memuat dataset DBLP untuk Role 2:
1. Memeriksa keberadaan data/clean_inproceedings.parquet (jika sudah ada hasil Task 1).
2. Jika belum ada tapi ada data/raw_inproceedings.parquet, menjalankan pipeline cleaning standar Task 1.
3. Jika belum ada file parquet di disk (karena file raw belum dicopy dari laptop Role 1),
   menghasilkan benchmark sample dataset (40.000 paper, 2015-2025) dengan pola kluster konferensi
   realistis sesuai ekosistem Computer Science DBLP agar seluruh kode Task 3, 4, 5, dan 6
   dapat langsung dieksekusi, diuji, dan menghasilkan metrik yang valid.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

DATA_DIR = Path("data")
CLEAN_PARQUET = DATA_DIR / "clean_inproceedings.parquet"
RAW_PARQUET = DATA_DIR / "raw_inproceedings.parquet"
SAMPLE_PARQUET = DATA_DIR / "clean_inproceedings_sample.parquet"

# Kluster Konferensi Realistis DBLP Computer Science
CONFERENCE_CLUSTERS = {
    "AI_ML": ["NeurIPS", "ICML", "ICLR", "AAAI", "IJCAI"],
    "Vision": ["CVPR", "ICCV", "ECCV"],
    "NLP": ["ACL", "EMNLP", "NAACL", "COLING"],
    "Data_IR": ["KDD", "SIGIR", "WSDM", "CIKM", "TheWebConf"],
    "Database": ["SIGMOD", "VLDB", "ICDE", "EDBT"],
    "Software_Eng": ["ICSE", "FSE", "ASE", "ISSTA"],
    "Systems_Net": ["OSDI", "SOSP", "SIGCOMM", "NSDI", "FAST"],
    "Security": ["IEEE S&P", "USENIX Security", "ACM CCS", "NDSS"],
    "Hardware_Arch": ["ISCA", "MICRO", "ASPLOS", "DAC"],
    "Theory": ["STOC", "FOCS", "SODA"]
}

def generate_benchmark_sample(n_papers=45_000, seed=42):
    """Menghasilkan sample dataset realistis DBLP untuk pengujian algoritma Mining."""
    np.random.seed(seed)
    DATA_DIR.mkdir(exist_ok=True)
    
    clusters = list(CONFERENCE_CLUSTERS.keys())
    cluster_weights = [0.22, 0.16, 0.12, 0.12, 0.10, 0.08, 0.07, 0.06, 0.04, 0.03]
    
    # 15.000 author dengan afiliasi domain riset utama
    n_authors = 15_000
    author_domains = np.random.choice(clusters, size=n_authors, p=cluster_weights)
    author_names = [f"Author_{i:05d}" for i in range(n_authors)]
    
    # Berikan nama-nama peneliti terkemuka DBLP untuk realisme
    famous_authors = [
        ("Yoshua Bengio", "AI_ML"), ("Geoffrey E. Hinton", "AI_ML"), ("Yann LeCun", "AI_ML"),
        ("Andrew Y. Ng", "AI_ML"), ("Michael I. Jordan", "AI_ML"), ("Bernhard Schölkopf", "AI_ML"),
        ("Kaiming He", "Vision"), ("Jitendra Malik", "Vision"), ("Trevor Darrell", "Vision"),
        ("Christopher D. Manning", "NLP"), ("Dan Jurafsky", "NLP"), ("Yoav Goldberg", "NLP"),
        ("Jiawei Han 0001", "Data_IR"), ("Philip S. Yu", "Data_IR"), ("Jure Leskovec", "Data_IR"),
        ("Michael Stonebraker", "Database"), ("Hector Garcia-Molina", "Database"), ("Gerhard Weikum", "Database"),
        ("David A. Patterson", "Hardware_Arch"), ("John L. Hennessy", "Hardware_Arch"),
        ("Ion Stoica", "Systems_Net"), ("Matei Zaharia", "Systems_Net"), ("Scott Shenker", "Systems_Net"),
        ("Dan Boneh", "Security"), ("Dawn Song", "Security"), ("Adi Shamir", "Security"),
        ("Martin Rinard", "Software_Eng"), ("Premkumar T. Devanbu", "Software_Eng")
    ]
    for idx, (name, dom) in enumerate(famous_authors):
        author_names[idx] = name
        author_domains[idx] = dom
        
    papers = []
    years = list(range(2015, 2026))
    year_weights = [0.06, 0.06, 0.07, 0.08, 0.09, 0.09, 0.10, 0.10, 0.11, 0.12, 0.12]
    
    # Buat pool author per domain
    domain_to_authors = {}
    for dom in clusters:
        domain_to_authors[dom] = [author_names[i] for i, d in enumerate(author_domains) if d == dom]
        
    for p_id in range(n_papers):
        # Pilih domain paper
        dom = np.random.choice(clusters, p=cluster_weights)
        venue = np.random.choice(CONFERENCE_CLUSTERS[dom])
        year = int(np.random.choice(years, p=year_weights))
        
        # 1-6 author per paper
        team_size = np.random.choice([1, 2, 3, 4, 5, 6], p=[0.08, 0.22, 0.32, 0.22, 0.11, 0.05])
        
        # 85% author dari domain yang sama, 15% kolaborator lintas bidang
        paper_authors = []
        for _ in range(team_size):
            if np.random.rand() < 0.85 or len(domain_to_authors[dom]) == 0:
                chosen = np.random.choice(domain_to_authors[dom])
            else:
                other_dom = np.random.choice(clusters)
                chosen = np.random.choice(domain_to_authors[other_dom])
            if chosen not in paper_authors:
                paper_authors.append(chosen)
                
        papers.append({
            "paper_key": f"conf/{venue.lower()}/{p_id}_{year}",
            "authors": paper_authors,
            "title": f"Advancements in {venue} Research: Paper #{p_id}",
            "year": year,
            "venue": venue
        })
        
    df_sample = pd.DataFrame(papers)
    df_sample.to_parquet(SAMPLE_PARQUET, index=False)
    print(f"[DataLoader] Generated sample dataset: {len(df_sample):,} records saved to {SAMPLE_PARQUET}")
    return df_sample

def load_clean_data():
    """
    Memuat dataset bersih:
    - Jika clean_inproceedings.parquet ada, gunakan langsung.
    - Jika raw_inproceedings.parquet ada, lakukan cleaning cepat Task 1 dan simpan.
    - Jika tidak ada, gunakan/buat sample dataset.
    """
    if CLEAN_PARQUET.exists():
        print(f"[DataLoader] Memuat dataset bersih dari: {CLEAN_PARQUET}")
        return pd.read_parquet(CLEAN_PARQUET)
        
    if RAW_PARQUET.exists():
        print(f"[DataLoader] Ditemukan raw data di {RAW_PARQUET}. Menjalankan cleaning Task 1...")
        df_raw = pd.read_parquet(RAW_PARQUET)
        
        # 1. Filter conference prefix
        df = df_raw[df_raw["key_prefix"] == "conf"].copy()
        
        # 2. Drop missing values
        df["authors"] = df["authors"].map(lambda L: [a.strip() for a in L if a and str(a).strip()])
        mask_valid = (
            (df["authors"].str.len() > 0) &
            (df["booktitle"].notna()) & (df["booktitle"].str.strip() != "") &
            (df["venue_key"].notna()) & (df["venue_key"].str.strip() != "") &
            (df["year_raw"].notna())
        )
        df = df[mask_valid].copy()
        
        # 3. Normalisasi Tahun
        df["year"] = pd.to_numeric(df["year_raw"], errors="coerce").astype("Int64")
        df = df[(df["year"] >= 2015) & (df["year"] <= 2025)].copy()
        df["year"] = df["year"].astype(int)
        
        # 4. Normalisasi Venue (standard uppercase acronym)
        # Utamakan booktitle yang sudah bersih, atau venue_key
        def clean_venue(row):
            bt = str(row["booktitle"]).strip()
            # Bersihkan imbuhan tahun/volume jika ada
            bt_clean = bt.split("'")[0].split("20")[0].strip()
            return bt_clean.upper() if len(bt_clean) >= 2 else str(row["venue_key"]).upper()
            
        df["venue"] = df.apply(clean_venue, axis=1)
        
        # 5. Drop duplikat persis
        df = df.drop_duplicates(subset=["paper_key"]).copy()
        
        # Simpan hasil clean
        df_out = df[["paper_key", "authors", "title", "year", "venue"]].copy()
        df_out.to_parquet(CLEAN_PARQUET, index=False)
        print(f"[DataLoader] Preprocessing selesai. Disimpan ke {CLEAN_PARQUET} ({len(df_out):,} baris)")
        return df_out
        
    if SAMPLE_PARQUET.exists():
        print(f"[DataLoader] Memuat benchmark sample dataset dari: {SAMPLE_PARQUET}")
        return pd.read_parquet(SAMPLE_PARQUET)
        
    print("[DataLoader] File raw_inproceedings.parquet belum ada di folder data/.")
    print("[DataLoader] Membuat benchmark sample dataset untuk pengujian Role 2...")
    return generate_benchmark_sample()

if __name__ == "__main__":
    df = load_clean_data()
    print("DataFrame Info:")
    print(df.info())
    print("\nSample records:")
    print(df.head(5))
