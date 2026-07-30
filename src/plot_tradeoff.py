"""
plot_tradeoff.py – 精度×多様性トレードオフのグラフを生成
output: report/plan_001/tradeoff.png
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── データ読み込み ────────────────────────────────────────────────
results_path = Path("report/plan_001/results.json")
with open(results_path) as f:
    data = json.load(f)

models = list(data.keys())
recall_cum   = [data[m]["mean"]["recall_cum"]       for m in models]
recall_single= [data[m]["mean"]["recall_single"]     for m in models]
overlap      = [data[m]["mean"]["temporal_overlap"]  for m in models]
coverage     = [data[m]["mean"]["coverage"]           for m in models]
ild          = [data[m]["mean"]["ild"]               for m in models]

recall_cum_std   = [data[m]["std"]["recall_cum"]      for m in models]
overlap_std      = [data[m]["std"]["temporal_overlap"] for m in models]

diversity = [1 - o for o in overlap]   # Temporal Diversity = 1 - Overlap

# ── カラーパレット ────────────────────────────────────────────────
colors = {
    "M0_baseline":        "#8e9aaf",
    "M1_clustering":      "#e63946",
    "M2_random_attention":"#457b9d",
    "M3_random_subset":   "#f4a261",
    "M4_gaussian_noise":  "#2a9d8f",
    "M5_mc_dropout":      "#9b5de5",
    "M6_vae":             "#b5838d",
}
short_labels = {
    "M0_baseline":        "M0\nBaseline",
    "M1_clustering":      "M1\nClustering",
    "M2_random_attention":"M2\nRandom\nAttn",
    "M3_random_subset":   "M3\nRandom\nSubset",
    "M4_gaussian_noise":  "M4\nGaussian\nNoise",
    "M5_mc_dropout":      "M5\nMC Dropout",
    "M6_vae":             "M6\nVAE",
}

fig = plt.figure(figsize=(16, 12), facecolor="#0f1117")
fig.suptitle(
    "Two-Tower Embedding Diversity: M0–M6 Comparison\nMovieLens 1M  |  K=10  |  N_trials=10  |  5 seeds",
    color="white", fontsize=15, fontweight="bold", y=0.98,
)

ax_color = "#1a1d27"
text_color = "#e0e0e0"
grid_color = "#2d3142"

# ── サブプロット1: recall_cum vs Temporal Diversity ───────────────
ax1 = fig.add_subplot(2, 2, 1, facecolor=ax_color)
for i, m in enumerate(models):
    c = colors[m]
    ax1.errorbar(
        diversity[i], recall_cum[i],
        xerr=overlap_std[i], yerr=recall_cum_std[i],
        fmt="o", color=c, markersize=12, ecolor=c, alpha=0.9,
        elinewidth=1.5, capsize=4, zorder=3,
    )
    # ラベルオフセット調整
    offsets = {
        "M0_baseline":        (-0.03, -0.0025),
        "M1_clustering":      (0.01, 0.0008),
        "M2_random_attention":(-0.02, 0.001),
        "M3_random_subset":   (0.01, -0.0025),
        "M4_gaussian_noise":  (0.005, 0.001),
        "M5_mc_dropout":      (0.01, 0.0005),
        "M6_vae":             (-0.04, -0.0025),
    }
    ox, oy = offsets.get(m, (0.01, 0.001))
    ax1.annotate(
        short_labels[m], xy=(diversity[i], recall_cum[i]),
        xytext=(diversity[i]+ox, recall_cum[i]+oy),
        color=c, fontsize=7.5, ha="left", va="center",
    )

# Pareto frontier
pts = sorted(zip(diversity, recall_cum), key=lambda x: x[0])
pareto = [pts[0]]
for d, r in pts[1:]:
    if r >= pareto[-1][1]:
        pareto.append((d, r))
if len(pareto) > 1:
    px, py = zip(*pareto)
    ax1.plot(px, py, "--", color="#ffd166", linewidth=1.2, alpha=0.6, label="Pareto frontier")
    ax1.legend(facecolor=ax_color, edgecolor=grid_color, labelcolor=text_color, fontsize=8)

ax1.set_xlabel("Temporal Diversity (1 − Overlap)  ↑", color=text_color, fontsize=10)
ax1.set_ylabel("Recall@10 Cumulative  ↑", color=text_color, fontsize=10)
ax1.set_title("Accuracy–Diversity Tradeoff", color=text_color, fontsize=11, pad=8)
ax1.tick_params(colors=text_color)
ax1.spines[:].set_color(grid_color)
ax1.grid(True, color=grid_color, linewidth=0.5, alpha=0.7)

# ── サブプロット2: Coverage vs Temporal Diversity ─────────────────
ax2 = fig.add_subplot(2, 2, 2, facecolor=ax_color)
for i, m in enumerate(models):
    c = colors[m]
    ax2.scatter(diversity[i], coverage[i], color=c, s=160, zorder=3, alpha=0.9)
    ax2.annotate(
        short_labels[m], (diversity[i], coverage[i]),
        textcoords="offset points", xytext=(6, 3),
        color=c, fontsize=7.5,
    )

ax2.set_xlabel("Temporal Diversity (1 − Overlap)  ↑", color=text_color, fontsize=10)
ax2.set_ylabel("Item Coverage  ↑", color=text_color, fontsize=10)
ax2.set_title("Coverage vs Temporal Diversity", color=text_color, fontsize=11, pad=8)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
ax2.tick_params(colors=text_color)
ax2.spines[:].set_color(grid_color)
ax2.grid(True, color=grid_color, linewidth=0.5, alpha=0.7)

# ── サブプロット3: 指標バーチャート (recall_cum, recall_single) ───
ax3 = fig.add_subplot(2, 2, 3, facecolor=ax_color)
x = np.arange(len(models))
width = 0.38
bars1 = ax3.bar(x - width/2, recall_cum,   width, color=[colors[m] for m in models], alpha=0.9, label="recall_cum")
bars2 = ax3.bar(x + width/2, recall_single, width, color=[colors[m] for m in models], alpha=0.4, label="recall_single", hatch="//")
ax3.errorbar(x - width/2, recall_cum, yerr=recall_cum_std,
             fmt="none", color="white", elinewidth=1.2, capsize=3)
ax3.set_xticks(x)
ax3.set_xticklabels([short_labels[m].replace("\n", " ") for m in models], rotation=30, ha="right", fontsize=7.5, color=text_color)
ax3.set_ylabel("Recall@10", color=text_color, fontsize=10)
ax3.set_title("recall_cum vs recall_single per Model", color=text_color, fontsize=11, pad=8)
ax3.tick_params(colors=text_color)
ax3.spines[:].set_color(grid_color)
ax3.grid(True, axis="y", color=grid_color, linewidth=0.5, alpha=0.7)
solid_patch = mpatches.Patch(facecolor="white", alpha=0.9, label="recall_cum (cumulative)")
hatch_patch = mpatches.Patch(facecolor="white", alpha=0.4, hatch="//", label="recall_single (per trial)")
ax3.legend(handles=[solid_patch, hatch_patch], facecolor=ax_color, edgecolor=grid_color,
           labelcolor=text_color, fontsize=8)

# ── サブプロット4: Temporal Overlap ホリゾンタルバー ─────────────
ax4 = fig.add_subplot(2, 2, 4, facecolor=ax_color)
y = np.arange(len(models))
hbars = ax4.barh(y, overlap, color=[colors[m] for m in models], alpha=0.85)
ax4.errorbar(overlap, y, xerr=overlap_std, fmt="none",
             color="white", elinewidth=1.2, capsize=3)
ax4.set_yticks(y)
ax4.set_yticklabels([short_labels[m].replace("\n", " ") for m in models],
                    fontsize=8, color=text_color)
ax4.set_xlabel("Temporal Overlap Rate  ↓ (lower = more diverse)", color=text_color, fontsize=10)
ax4.set_title("Temporal Overlap Rate by Model", color=text_color, fontsize=11, pad=8)
ax4.axvline(1.0, color="#ffd166", linewidth=1, linestyle="--", alpha=0.5)
for bar, val in zip(hbars, overlap):
    ax4.text(val + 0.01, bar.get_y() + bar.get_height()/2,
             f"{val:.3f}", va="center", color=text_color, fontsize=8)
ax4.tick_params(colors=text_color)
ax4.spines[:].set_color(grid_color)
ax4.grid(True, axis="x", color=grid_color, linewidth=0.5, alpha=0.7)
ax4.set_xlim(0, 1.12)

plt.tight_layout(rect=[0, 0, 1, 0.96])
out = Path("report/plan_001/tradeoff.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {out}")
plt.close()
