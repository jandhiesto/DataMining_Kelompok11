"""
Module src/data_loader.py
Loader terverifikasi untuk dataset bersih hasil Task 1 (Role 1 — Notebook-dblp_2025_12_01.ipynb).

Fitur utama:
1. Membaca `data/clean_inproceedings.parquet` (hanya kolom yang dibutuhkan Role 2 agar hemat RAM).
2. Memverifikasi ukuran berkas, jumlah baris, dan checksum SHA-256 terhadap `data/clean_summary.json`
   agar dipastikan berkas yang dipakai berasal dari folder `data_baru/` (bukan parquet versi lama).
3. Membangun peta label venue 1-ke-1 (`venue_key -> display_label`) yang menangani kemungkinan
   tabrakan `venue_name` antar `venue_key` berbeda (selaras dengan keputusan T04 Role 1).
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

DATA_DIR = Path("data")
CLEAN_PARQUET = DATA_DIR / "clean_inproceedings.parquet"
CLEAN_SUMMARY_JSON = DATA_DIR / "clean_summary.json"

REQUIRED_COLS = ["paper_key", "author_id", "year", "venue_key", "venue_name"]


def compute_sha256(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    """Menghitung SHA-256 berkas secara streaming (hemat RAM)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block_size):
            h.update(chunk)
    return h.hexdigest()


def load_clean_summary(summary_path: Path = CLEAN_SUMMARY_JSON) -> Optional[dict]:
    """Membaca metadata `clean_summary.json` hasil ekspor T06 Role 1 jika tersedia."""
    if summary_path.is_file():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    return None


def verify_clean_parquet(
    parquet_path: Path = CLEAN_PARQUET,
    summary_path: Path = CLEAN_SUMMARY_JSON,
    verify_hash: bool = True,
) -> dict:
    """
    Memastikan `clean_inproceedings.parquet` ada dan sesuai dengan `clean_summary.json`.
    Mengembalikan ringkasan status verifikasi.
    """
    if not parquet_path.is_file():
        raise FileNotFoundError(
            f"Berkas dataset bersih tidak ditemukan di: {parquet_path.resolve()}\n"
            "Langkah perbaikan:\n"
            "  1. Buka Google Drive tim -> masuk ke folder `data_baru/`.\n"
            "  2. Unduh `clean_inproceedings.parquet` (~205 MB).\n"
            f"  3. Simpan ke: {parquet_path.resolve()}"
        )

    size_bytes = parquet_path.stat().st_size
    summary = load_clean_summary(summary_path)

    info = {
        "path": str(parquet_path),
        "size_bytes": size_bytes,
        "sha256": None,
        "expected_sha256": summary.get("clean_parquet_sha256") if summary else None,
        "hash_match": None,
        "expected_rows": summary.get("n_papers_clean") if summary else None,
        "expected_unique_authors": summary.get("n_unique_authors") if summary else None,
        "expected_unique_venues": summary.get("n_unique_venues_all") if summary else None,
    }

    if verify_hash:
        actual_sha = compute_sha256(parquet_path)
        info["sha256"] = actual_sha
        if info["expected_sha256"]:
            info["hash_match"] = actual_sha == info["expected_sha256"]
            if not info["hash_match"]:
                raise ValueError(
                    "PERINGATAN: SHA-256 `clean_inproceedings.parquet` TIDAK COCOK dengan `data/clean_summary.json`!\n"
                    f"  Aktual     : {actual_sha} ({size_bytes:,} bytes)\n"
                    f"  Seharusnya : {info['expected_sha256']} ({summary.get('clean_parquet_size_bytes', 0):,} bytes)\n"
                    "Kemungkinan Anda mengunduh `clean_inproceedings.parquet` versi lama (di luar folder `data_baru/`).\n"
                    "Pastikan Anda mengunduh berkas dari dalam folder `data_baru/`!"
                )

    return info


def build_venue_label_map(df: pd.DataFrame) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Membangun pemetaan:
    - `venue_name_map`: `venue_key -> venue_name` (asli dari T04 Role 1).
    - `venue_display_map`: `venue_key -> label tampil bebas ambiguitas`.
      Jika ada dua `venue_key` berbeda yang memiliki `venue_name` sama (mis. `ICC`),
      label didisambiguasi menjadi `venue_name [venue_key]` agar tidak tertukar di tabel/grafik.
    """
    pairs = df[["venue_key", "venue_name"]].drop_duplicates().reset_index(drop=True)
    assert int(pairs.groupby("venue_key")["venue_name"].nunique().max()) == 1, (
        "Pemetaan venue_key -> venue_name harus 1-ke-1!"
    )

    venue_name_map = dict(zip(pairs["venue_key"], pairs["venue_name"]))
    name_freq = pairs["venue_name"].value_counts()
    dup_names = set(name_freq[name_freq > 1].index)

    venue_display_map = {
        vk: (f"{vn} [{vk}]" if vn in dup_names else vn)
        for vk, vn in venue_name_map.items()
    }
    return venue_name_map, venue_display_map


def load_clean_data(
    parquet_path: Path = CLEAN_PARQUET,
    summary_path: Path = CLEAN_SUMMARY_JSON,
    columns: Optional[List[str]] = None,
    verify_hash: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """
    Memuat `clean_inproceedings.parquet` beserta hasil verifikasi integritasnya.
    Default hanya membaca kolom yang relevan untuk Role 2 agar hemat RAM.
    """
    info = verify_clean_parquet(parquet_path, summary_path, verify_hash=verify_hash)

    cols_to_read = columns if columns is not None else [
        "paper_key",
        "author_id",
        "year",
        "venue_key",
        "venue_name",
        "in_conf_prefix",
        "is_workshop",
    ]
    df = pd.read_parquet(parquet_path, columns=cols_to_read)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Kolom wajib tidak ditemukan pada parquet: {missing}. "
            f"Kolom tersedia: {list(df.columns)}"
        )

    if info["expected_rows"] is not None:
        assert len(df) == info["expected_rows"], (
            f"Jumlah baris ({len(df):,}) tidak sesuai clean_summary.json ({info['expected_rows']:,})"
        )

    print(f"[DataLoader] PASS - Memuat {parquet_path} ({len(df):,} publikasi bersih)")
    if info["hash_match"]:
        print(f"[DataLoader] PASS - SHA-256 terverifikasi cocok dengan data/clean_summary.json")
    return df, info


if __name__ == "__main__":
    df_clean, meta = load_clean_data()
    print(df_clean.info())
    print(df_clean.head(5))
