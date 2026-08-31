"""
Plot: Model Family & Model Size Scaling Overview (Refined Layout)
----------------------------------------------------------------
Visualizes how model families (mE5, Qwen3, F2LLM) and model sizes (0.1B to 8B params)
impact recommendation accuracy, ranking precision, and catalog diversity.
Includes cleanly offset annotations to avoid overlap.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = [
    # mE5 family
    {"model": "mE5-small", "family": "mE5 (RoBERTa)", "params_b": 0.117, "dim": 384,
     "recall": 0.027076, "recall_std": 0.000676, "ndcg": 0.022752, "ndcg_std": 0.000645,
     "prec": 1.6335, "prec_std": 0.0385, "coverage": 3.4252, "coverage_std": 0.3541, "longtail": 0.8167, "longtail_std": 0.1808},
    {"model": "mE5-base", "family": "mE5 (RoBERTa)", "params_b": 0.278, "dim": 768,
     "recall": 0.028309, "recall_std": 0.001021, "ndcg": 0.023659, "ndcg_std": 0.000833,
     "prec": 1.7186, "prec_std": 0.0594, "coverage": 4.2390, "coverage_std": 0.7108, "longtail": 1.0675, "longtail_std": 0.2566},
    {"model": "mE5-large", "family": "mE5 (RoBERTa)", "params_b": 0.560, "dim": 1024,
     "recall": 0.028466, "recall_std": 0.000991, "ndcg": 0.023924, "ndcg_std": 0.000924,
     "prec": 1.7457, "prec_std": 0.0477, "coverage": 5.1610, "coverage_std": 0.5146, "longtail": 1.6270, "longtail_std": 0.2781},

    # Qwen3 family
    {"model": "Qwen3-0.6B", "family": "Qwen3 (Official LLM)", "params_b": 0.59, "dim": 1024,
     "recall": 0.029368, "recall_std": 0.000820, "ndcg": 0.024632, "ndcg_std": 0.000760,
     "prec": 1.7966, "prec_std": 0.0546, "coverage": 3.4252, "coverage_std": 0.3763, "longtail": 0.7974, "longtail_std": 0.2942},
    {"model": "Qwen3-4B", "family": "Qwen3 (Official LLM)", "params_b": 4.02, "dim": 2560,
     "recall": 0.029059, "recall_std": 0.000330, "ndcg": 0.024752, "ndcg_std": 0.000261,
     "prec": 1.7990, "prec_std": 0.0206, "coverage": 3.8218, "coverage_std": 0.2190, "longtail": 0.9646, "longtail_std": 0.1345},
    {"model": "Qwen3-8B", "family": "Qwen3 (Official LLM)", "params_b": 8.04, "dim": 4096,
     "recall": 0.029127, "recall_std": 0.000355, "ndcg": 0.024695, "ndcg_std": 0.000194,
     "prec": 1.7901, "prec_std": 0.0095, "coverage": 4.5171, "coverage_std": 0.3959, "longtail": 1.3633, "longtail_std": 0.3013},

    # F2LLM family
    {"model": "F2LLM-0.6B", "family": "F2LLM (CodeFuse Fine-Tuned)", "params_b": 0.59, "dim": 1024,
     "recall": 0.028974, "recall_std": 0.001078, "ndcg": 0.024084, "ndcg_std": 0.001349,
     "prec": 1.7547, "prec_std": 0.0757, "coverage": 3.2501, "coverage_std": 0.2794, "longtail": 0.6302, "longtail_std": 0.2246},
    {"model": "F2LLM-4B", "family": "F2LLM (CodeFuse Fine-Tuned)", "params_b": 4.02, "dim": 2560,
     "recall": 0.028568, "recall_std": 0.000489, "ndcg": 0.024453, "ndcg_std": 0.000432,
     "prec": 1.7681, "prec_std": 0.0302, "coverage": 4.3420, "coverage_std": 0.5599, "longtail": 1.3505, "longtail_std": 0.3729},
    {"model": "F2LLM-8B", "family": "F2LLM (CodeFuse Fine-Tuned)", "params_b": 8.04, "dim": 4096,
     "recall": 0.028971, "recall_std": 0.000678, "ndcg": 0.024823, "ndcg_std": 0.000471,
     "prec": 1.7932, "prec_std": 0.0315, "coverage": 4.1102, "coverage_std": 0.1808, "longtail": 1.0932, "longtail_std": 0.2352},
]

df_all = pd.DataFrame(data)

# Styling configuration
plt.rcParams['font.family'] = 'sans-serif'
fig, axes = plt.subplots(2, 3, figsize=(21, 13))
fig.suptitle("Scaling Analysis: Model Family vs. Model Size (0.1B - 8B Parameters)\n(MovieLens 1M: 5 Seeds Mean ± Std Error Bars)",
             fontsize=17, fontweight='bold', y=0.98)

family_styles = {
    "mE5 (RoBERTa)": {"color": "#5d6d7e", "marker": "o", "ls": "--", "label": "mE5 (RoBERTa Baseline)"},
    "Qwen3 (Official LLM)": {"color": "#d35400", "marker": "s", "ls": "-", "label": "Qwen3 (Official LLM)"},
    "F2LLM (CodeFuse Fine-Tuned)": {"color": "#8e44ad", "marker": "^", "ls": "-.", "label": "F2LLM (CodeFuse Fine-Tuned)"},
}

def get_offset(fam: str, metric: str = "recall"):
    if "Qwen3" in fam:
        return (0, 10), "bottom"
    elif "F2LLM" in fam:
        return (0, -15), "top"
    else:
        return (-10, 8), "bottom"

# (1) Recall@10 vs. Model Size (Log Scale)
ax1 = axes[0, 0]
for fam, style in family_styles.items():
    sub = df_all[df_all["family"] == fam].sort_values("params_b")
    ax1.errorbar(
        sub["params_b"], sub["recall"], yerr=sub["recall_std"],
        label=style["label"], color=style["color"], marker=style["marker"],
        linestyle=style["ls"], linewidth=2.2, markersize=8.5, capsize=4.5, elinewidth=1.5
    )
    for _, row in sub.iterrows():
        xytext, va = get_offset(fam, "recall")
        ax1.annotate(f"{row['model']}\n({row['dim']}d)", (row["params_b"], row["recall"]),
                     textcoords="offset points", xytext=xytext, ha='center', va=va, fontsize=8.2, fontweight='bold', color=style["color"])

ax1.set_xscale("log")
ax1.set_ylim(0.0260, 0.0305)
ax1.set_title("(a) Recommendation Accuracy: Recall@10 vs. Model Size ↑", fontweight='bold', fontsize=11.5)
ax1.set_xlabel("Model Parameters [Billion] (Log Scale)", fontsize=10.5)
ax1.set_ylabel("Recall@10", fontsize=10.5)
ax1.grid(True, alpha=0.3, which='both')
ax1.legend(loc='lower right', fontsize=8.5)

# (2) NDCG@10 vs. Model Size (Log Scale)
ax2 = axes[0, 1]
for fam, style in family_styles.items():
    sub = df_all[df_all["family"] == fam].sort_values("params_b")
    ax2.errorbar(
        sub["params_b"], sub["ndcg"], yerr=sub["ndcg_std"],
        label=style["label"], color=style["color"], marker=style["marker"],
        linestyle=style["ls"], linewidth=2.2, markersize=8.5, capsize=4.5, elinewidth=1.5
    )
    for _, row in sub.iterrows():
        xytext, va = get_offset(fam, "ndcg")
        ax2.annotate(f"{row['model']}", (row["params_b"], row["ndcg"]),
                     textcoords="offset points", xytext=xytext, ha='center', va=va, fontsize=8.2, fontweight='bold', color=style["color"])

ax2.set_xscale("log")
ax2.set_ylim(0.0218, 0.0257)
ax2.set_title("(b) Ranking Quality: NDCG@10 vs. Model Size ↑", fontweight='bold', fontsize=11.5)
ax2.set_xlabel("Model Parameters [Billion] (Log Scale)", fontsize=10.5)
ax2.set_ylabel("NDCG@10", fontsize=10.5)
ax2.grid(True, alpha=0.3, which='both')
ax2.legend(loc='lower right', fontsize=8.5)

# (3) Precision@10 (%) vs. Model Size (Log Scale)
ax3 = axes[0, 2]
for fam, style in family_styles.items():
    sub = df_all[df_all["family"] == fam].sort_values("params_b")
    ax3.errorbar(
        sub["params_b"], sub["prec"], yerr=sub["prec_std"],
        label=style["label"], color=style["color"], marker=style["marker"],
        linestyle=style["ls"], linewidth=2.2, markersize=8.5, capsize=4.5, elinewidth=1.5
    )
    for _, row in sub.iterrows():
        xytext, va = get_offset(fam, "prec")
        ax3.annotate(f"{row['model']}", (row["params_b"], row["prec"]),
                     textcoords="offset points", xytext=xytext, ha='center', va=va, fontsize=8.2, fontweight='bold', color=style["color"])

ax3.set_xscale("log")
ax3.set_ylim(1.58, 1.87)
ax3.set_title("(c) Precision@10 (%) vs. Model Size ↑", fontweight='bold', fontsize=11.5)
ax3.set_xlabel("Model Parameters [Billion] (Log Scale)", fontsize=10.5)
ax3.set_ylabel("Precision@10 (%)", fontsize=10.5)
ax3.grid(True, alpha=0.3, which='both')
ax3.legend(loc='lower right', fontsize=8.5)

# (4) Catalog Coverage@10 (%) vs. Model Size (Log Scale)
ax4 = axes[1, 0]
for fam, style in family_styles.items():
    sub = df_all[df_all["family"] == fam].sort_values("params_b")
    ax4.errorbar(
        sub["params_b"], sub["coverage"], yerr=sub["coverage_std"],
        label=style["label"], color=style["color"], marker=style["marker"],
        linestyle=style["ls"], linewidth=2.2, markersize=8.5, capsize=4.5, elinewidth=1.5
    )
    for _, row in sub.iterrows():
        xytext = (0, 10) if "mE5" in fam or "F2LLM" in fam else (0, -14)
        va = "bottom" if "mE5" in fam or "F2LLM" in fam else "top"
        ax4.annotate(f"{row['model']}", (row["params_b"], row["coverage"]),
                     textcoords="offset points", xytext=xytext, ha='center', va=va, fontsize=8.2, fontweight='bold', color=style["color"])

ax4.set_xscale("log")
ax4.set_ylim(2.8, 5.8)
ax4.set_title("(d) Catalog Coverage@10 (%) vs. Model Size", fontweight='bold', fontsize=11.5)
ax4.set_xlabel("Model Parameters [Billion] (Log Scale)", fontsize=10.5)
ax4.set_ylabel("Catalog Coverage@10 (%)", fontsize=10.5)
ax4.grid(True, alpha=0.3, which='both')
ax4.legend(loc='upper left', fontsize=8.5)

# (5) Long-tail Coverage (%) vs. Model Size (Log Scale)
ax5 = axes[1, 1]
for fam, style in family_styles.items():
    sub = df_all[df_all["family"] == fam].sort_values("params_b")
    ax5.errorbar(
        sub["params_b"], sub["longtail"], yerr=sub["longtail_std"],
        label=style["label"], color=style["color"], marker=style["marker"],
        linestyle=style["ls"], linewidth=2.2, markersize=8.5, capsize=4.5, elinewidth=1.5
    )
    for _, row in sub.iterrows():
        xytext = (0, 10) if "mE5" in fam or "F2LLM" in fam else (0, -14)
        va = "bottom" if "mE5" in fam or "F2LLM" in fam else "top"
        ax5.annotate(f"{row['model']}", (row["params_b"], row["longtail"]),
                     textcoords="offset points", xytext=xytext, ha='center', va=va, fontsize=8.2, fontweight='bold', color=style["color"])

ax5.set_xscale("log")
ax5.set_ylim(0.3, 2.0)
ax5.set_title("(e) Long-tail Item Coverage (%) vs. Model Size", fontweight='bold', fontsize=11.5)
ax5.set_xlabel("Model Parameters [Billion] (Log Scale)", fontsize=10.5)
ax5.set_ylabel("Long-tail Coverage (%)", fontsize=10.5)
ax5.grid(True, alpha=0.3, which='both')
ax5.legend(loc='upper left', fontsize=8.5)

# (6) Comprehensive Pareto Trade-off: Recall@10 vs. Catalog Coverage@10 (%)
ax6 = axes[1, 2]
pareto_offsets = {
    "mE5-small": (-12, -14),
    "mE5-base": (8, -10),
    "mE5-large": (8, 0),
    "Qwen3-0.6B": (8, 6),
    "Qwen3-4B": (8, -6),
    "Qwen3-8B": (8, 6),
    "F2LLM-0.6B": (-12, -14),
    "F2LLM-4B": (8, -12),
    "F2LLM-8B": (-15, 8),
}

for _, row in df_all.iterrows():
    fam = row["family"]
    style = family_styles[fam]
    ax6.errorbar(
        row["coverage"], row["recall"],
        xerr=row["coverage_std"], yerr=row["recall_std"],
        fmt=style["marker"], color=style["color"], ecolor=style["color"],
        elinewidth=1.8, capsize=4, markersize=8.5, alpha=0.9
    )
    offset = pareto_offsets.get(row["model"], (6, 5))
    ax6.annotate(f"{row['model']}", (row["coverage"], row["recall"]),
                 textcoords="offset points", xytext=offset, fontsize=8.2, fontweight='bold', color=style["color"])

# Dummy points for Pareto legend
for fam, style in family_styles.items():
    ax6.plot([], [], marker=style["marker"], color=style["color"], linestyle='None',
             label=style["label"], markersize=8.5)

ax6.set_title("(f) Pareto Frontier: Recall@10 vs. Catalog Coverage@10 (%)", fontweight='bold', fontsize=11.5)
ax6.set_xlabel("Catalog Coverage@10 (%) [Higher Diversity →]", fontsize=10.5)
ax6.set_ylabel("Recall@10 [Higher Accuracy →]", fontsize=10.5)
ax6.grid(True, alpha=0.3)
ax6.legend(loc='lower left', fontsize=8.5)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

out_paths = [
    Path("report/model_family_scaling_overview.png"),
    Path("report/plan_022/family_size_scaling_overview.png"),
]
for p in out_paths:
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p, dpi=300)
    print(f"Saved figure to {p}")

plt.close()
