import json
import io
import sys
import base64
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

NOTEBOOK_PATH = Path("ProjectDataMining_dblpDec2025.ipynb")

with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Global namespace untuk eksekusi
ns = {
    "pd": pd,
    "np": np,
    "plt": plt,
    "Path": Path,
    "OUT_DIR": Path("data"),
    "RAW_PARQUET": Path("data/raw_inproceedings.parquet"),
    "COUNTS_JSON": Path("data/raw_counts.json")
}

# Dummy display function
def notebook_display(*args):
    for a in args:
        if isinstance(a, pd.DataFrame):
            print(a.to_string())
        else:
            print(a)

ns["display"] = notebook_display

print("Mulai mengeksekusi sel-sel Role 2...")

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code" or i < 20:
        continue
    
    src = "".join(cell["source"])
    print(f"Executing Cell {i}...")
    
    # Tangkap stdout
    old_stdout = sys.stdout
    captured_out = io.StringIO()
    sys.stdout = captured_out
    
    # Tangkap plot jika ada
    plt.close('all')
    
    try:
        exec(src, ns)
    except Exception as e:
        sys.stdout = old_stdout
        print(f"Error in Cell {i}: {e}")
        break
    finally:
        sys.stdout = old_stdout
        
    out_text = captured_out.getvalue()
    
    # Siapkan output cell
    cell_outputs = []
    if out_text.strip():
        cell_outputs.append({
            "name": "stdout",
            "output_type": "stream",
            "text": [line + "\n" for line in out_text.splitlines()]
        })
        
    # Cek apakah ada plot yang digambar
    figs = [plt.figure(n) for n in plt.get_fignums()]
    if figs:
        for fig in figs:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode("utf-8")
            cell_outputs.append({
                "data": {
                    "image/png": img_b64,
                    "text/plain": ["<Figure size ...>"]
                },
                "metadata": {},
                "output_type": "display_data"
            })
        plt.close('all')
        
    cell["outputs"] = cell_outputs
    cell["execution_count"] = i

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Semua sel berhasil dieksekusi dan outputnya tersimpan langsung di dalam notebook!")
