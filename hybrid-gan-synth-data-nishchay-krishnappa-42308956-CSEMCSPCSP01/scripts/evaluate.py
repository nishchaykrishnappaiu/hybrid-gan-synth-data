import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import ks_2samp
from scripts.utils import detect_column_types

def evaluate(real_csv, synth_csv, plots_dir):
    real = pd.read_csv(real_csv)
    synth = pd.read_csv(synth_csv)

    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    categorical, continuous = detect_column_types(real)

    report_lines = []

    for col in real.columns:
        r = real[col]
        s = synth[col]

        # CONTINUOUS COLUMNS
        if col in continuous:
            r_num = pd.to_numeric(r, errors="coerce").dropna()
            s_num = pd.to_numeric(s, errors="coerce").dropna()

            if len(r_num) == 0 or len(s_num) == 0:
                continue

            ks, p = ks_2samp(r_num, s_num)
            report_lines.append(f"CONTINUOUS {col}: KS={ks:.4f}, p={p:.4f}")

            plt.figure(figsize=(10, 4))
            plt.hist(r_num, bins=50, alpha=0.6, label="Real")
            plt.hist(s_num, bins=50, alpha=0.6, label="Synthetic")
            plt.title(f"Histogram - {col}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{plots_dir}/hist_{col}.png")
            plt.close()

        # CATEGORICAL COLUMNS
        else:
            r_counts = r.value_counts(normalize=True)
            s_counts = s.value_counts(normalize=True)

            cats = sorted(set(r_counts.index).union(set(s_counts.index)))

            r_vals = np.array([r_counts.get(c, 0) for c in cats])
            s_vals = np.array([s_counts.get(c, 0) for c in cats])

            l1 = np.abs(r_vals - s_vals).sum()

            report_lines.append(f"CATEGORICAL {col}: L1 distance={l1:.4f}")

            plt.figure(figsize=(12, 4))
            plt.bar(range(len(cats)), r_vals, width=0.4, label="Real")
            plt.bar([x + 0.4 for x in range(len(cats))], s_vals, width=0.4, label="Synthetic")
            plt.xticks(range(len(cats)), cats, rotation=90)
            plt.title(f"Category Dist - {col}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{plots_dir}/bar_{col}.png")
            plt.close()

    with open(f"{plots_dir}/evaluation_report.txt", "w") as f:
        f.write("\n".join(report_lines))

    print("✅ Evaluation Complete. Check plots folder.")
