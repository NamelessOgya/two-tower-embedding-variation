import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set matplotlib style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 1.0

# Create output dir
os.makedirs('report', exist_ok=True)

df_ml = pd.read_csv('report/plan_016/results_016.csv')
df_yelp = pd.read_csv('report/plan_017/results_yelp.csv')

# MovieLens data points
base_ml = df_ml[df_ml['model'] == 'TwoTower_d2_h64'].iloc[0]
part_ml = df_ml[df_ml['model'] == 'TT_item_partition_n10'].iloc[0]
strat_ml = df_ml[df_ml['model'] == 'TT_semantic_stratified_partition_n10'].iloc[0]
jaccard_ml = df_ml[df_ml['model'] == 'TT_divloss_soft_jaccard_l0p1_s0p05'].iloc[0]

pl_ml = df_ml[df_ml['model'].str.contains('TT_2stage_PL')].copy()
pl_ml['tau'] = [0.2, 0.5, 1.0, 2.0, 5.0]
pl_ml = pl_ml.sort_values('tau')

noise_div_ml = [0.0, 0.293, 0.706, 0.894]
noise_prec_ml = [0.852, 0.835, 0.780, 0.605]
noise_recall_ml = [0.0093, 0.0188, 0.0404, 0.0529]

# Yelp data points
base_y = df_yelp[df_yelp['model'] == 'TwoTower_d2_h64'].iloc[0]
part_y = df_yelp[df_yelp['model'] == 'TT_item_partition_n10'].iloc[0]
strat_y = df_yelp[df_yelp['model'] == 'TT_semantic_stratified_partition_n10'].iloc[0]
jaccard_y = df_yelp[df_yelp['model'] == 'TT_divloss_soft_jaccard_l0p1_s0p05'].iloc[0]

pl_y = df_yelp[df_yelp['model'].str.contains('TT_2stage_PL')].copy()
pl_y['tau'] = [1.0, 2.0, 5.0]
pl_y = pl_y.sort_values('tau')

noise_div_y = [0.0, 0.721]
noise_prec_y = [0.022, 0.017]
noise_recall_y = [0.0010, 0.0035]

# -------------------------------------------------------------
# 1. Total Slate Precision Comparison Plot (Cross-Dataset)
# -------------------------------------------------------------
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

# ML Precision
ax1.plot(pl_ml['diversity'], pl_ml['slate_precision_pct'], 'o-', color='#1f77b4', linewidth=2.5, markersize=8, label='2-Stage Plackett-Luce (M=200, tau sweep)', zorder=5)
for _, row in pl_ml.iterrows():
    ax1.annotate(f"tau={row['tau']}", (row['diversity'], row['slate_precision_pct']), textcoords="offset points", xytext=(8,-4), ha='left', fontsize=9, fontweight='bold', color='#1f77b4')

ax1.plot(noise_div_ml, noise_prec_ml, 's--', color='#d62728', linewidth=2.0, markersize=7, label='Simple Output Noise (sigma sweep)', zorder=4)
ax1.annotate('sigma=0.05', (noise_div_ml[2], noise_prec_ml[2]), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=9, color='#d62728')
ax1.annotate('sigma=0.10', (noise_div_ml[3], noise_prec_ml[3]), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=9, color='#d62728')

ax1.scatter([base_ml['diversity']], [base_ml['slate_precision_pct']], color='#7f7f7f', s=120, marker='D', label='Base Two-Tower (No Div)', zorder=6)
ax1.scatter([part_ml['diversity']], [part_ml['slate_precision_pct']], color='#2ca02c', s=120, marker='^', label='Random Item Partition (n=10)', zorder=6)
ax1.scatter([strat_ml['diversity']], [strat_ml['slate_precision_pct']], color='#9467bd', s=120, marker='v', label='Semantic Stratified Partition (n=10)', zorder=6)
ax1.scatter([jaccard_ml['diversity']], [jaccard_ml['slate_precision_pct']], color='#ff7f0e', s=120, marker='P', label='Soft-Jaccard DivLoss', zorder=6)

ax1.set_title('MovieLens 1M: Diversity vs Total Slate Precision (%)', fontsize=12, fontweight='bold', pad=12)
ax1.set_xlabel('Diversity (1 - Overlap)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Total Slate Precision (%)', fontsize=11, fontweight='bold')
ax1.legend(loc='upper left', fontsize=8.5, frameon=True, facecolor='white', framealpha=0.95)
ax1.set_ylim(0.50, 1.40)

# Yelp Precision
ax2.plot(pl_y['diversity'], pl_y['slate_precision_pct'], 'o-', color='#1f77b4', linewidth=2.5, markersize=8, label='2-Stage Plackett-Luce (M=200, tau sweep)', zorder=5)
for _, row in pl_y.iterrows():
    ax2.annotate(f"tau={row['tau']}", (row['diversity'], row['slate_precision_pct']), textcoords="offset points", xytext=(8,-4), ha='left', fontsize=9, fontweight='bold', color='#1f77b4')

ax2.plot(noise_div_y, noise_prec_y, 's--', color='#d62728', linewidth=2.0, markersize=7, label='Simple Output Noise (sigma sweep)', zorder=4)
ax2.annotate('sigma=0.05', (noise_div_y[1], noise_prec_y[1]), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=9, color='#d62728')

ax2.scatter([base_y['diversity']], [base_y['slate_precision_pct']], color='#7f7f7f', s=120, marker='D', label='Base Two-Tower (No Div)', zorder=6)
ax2.scatter([part_y['diversity']], [part_y['slate_precision_pct']], color='#2ca02c', s=120, marker='^', label='Random Item Partition (n=10)', zorder=6)
ax2.scatter([strat_y['diversity']], [strat_y['slate_precision_pct']], color='#9467bd', s=120, marker='v', label='Semantic Stratified Partition (n=10)', zorder=6)
ax2.scatter([jaccard_y['diversity']], [jaccard_y['slate_precision_pct']], color='#ff7f0e', s=120, marker='P', label='Soft-Jaccard DivLoss', zorder=6)

ax2.set_title('Yelp 10-Core: Diversity vs Total Slate Precision (%)', fontsize=12, fontweight='bold', pad=12)
ax2.set_xlabel('Diversity (1 - Overlap)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Total Slate Precision (%)', fontsize=11, fontweight='bold')
ax2.legend(loc='upper left', fontsize=8.5, frameon=True, facecolor='white', framealpha=0.95)
ax2.set_ylim(0.010, 0.048)

fig1.tight_layout()
fig1.savefig('report/tradeoff_comparison_precision.png', dpi=300, bbox_inches='tight')
plt.close(fig1)

# -------------------------------------------------------------
# 2. Total Recall Comparison Plot (Cross-Dataset)
# -------------------------------------------------------------
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

# ML Recall
ax3.plot(pl_ml['diversity'], pl_ml['recall_cum'], 'o-', color='#1f77b4', linewidth=2.5, markersize=8, label='2-Stage Plackett-Luce (M=200, tau sweep)', zorder=5)
for _, row in pl_ml.iterrows():
    ax3.annotate(f"tau={row['tau']}", (row['diversity'], row['recall_cum']), textcoords="offset points", xytext=(8,-4), ha='left', fontsize=9, fontweight='bold', color='#1f77b4')

ax3.plot(noise_div_ml, noise_recall_ml, 's--', color='#d62728', linewidth=2.0, markersize=7, label='Simple Output Noise (sigma sweep)', zorder=4)
ax3.annotate('sigma=0.05', (noise_div_ml[2], noise_recall_ml[2]), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=9, color='#d62728')
ax3.annotate('sigma=0.10', (noise_div_ml[3], noise_recall_ml[3]), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=9, color='#d62728')

ax3.scatter([base_ml['diversity']], [base_ml['recall_cum']], color='#7f7f7f', s=120, marker='D', label='Base Two-Tower (No Div)', zorder=6)
ax3.scatter([part_ml['diversity']], [part_ml['recall_cum']], color='#2ca02c', s=120, marker='^', label='Random Item Partition (n=10)', zorder=6)
ax3.scatter([strat_ml['diversity']], [strat_ml['recall_cum']], color='#9467bd', s=120, marker='v', label='Semantic Stratified Partition (n=10)', zorder=6)
ax3.scatter([jaccard_ml['diversity']], [jaccard_ml['recall_cum']], color='#ff7f0e', s=120, marker='P', label='Soft-Jaccard DivLoss', zorder=6)

ax3.set_title('MovieLens 1M: Diversity vs Total Recall (recall_cum)', fontsize=12, fontweight='bold', pad=12)
ax3.set_xlabel('Diversity (1 - Overlap)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Total Recall (10-Trial Cumulative Recall)', fontsize=11, fontweight='bold')
ax3.legend(loc='upper left', fontsize=8.5, frameon=True, facecolor='white', framealpha=0.95)
ax3.set_ylim(0.00, 0.15)

# Yelp Recall
ax4.plot(pl_y['diversity'], pl_y['recall_cum'], 'o-', color='#1f77b4', linewidth=2.5, markersize=8, label='2-Stage Plackett-Luce (M=200, tau sweep)', zorder=5)
for _, row in pl_y.iterrows():
    ax4.annotate(f"tau={row['tau']}", (row['diversity'], row['recall_cum']), textcoords="offset points", xytext=(8,-4), ha='left', fontsize=9, fontweight='bold', color='#1f77b4')

ax4.plot(noise_div_y, noise_recall_y, 's--', color='#d62728', linewidth=2.0, markersize=7, label='Simple Output Noise (sigma sweep)', zorder=4)
ax4.annotate('sigma=0.05', (noise_div_y[1], noise_recall_y[1]), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=9, color='#d62728')

ax4.scatter([base_y['diversity']], [base_y['recall_cum']], color='#7f7f7f', s=120, marker='D', label='Base Two-Tower (No Div)', zorder=6)
ax4.scatter([part_y['diversity']], [part_y['recall_cum']], color='#2ca02c', s=120, marker='^', label='Random Item Partition (n=10)', zorder=6)
ax4.scatter([strat_y['diversity']], [strat_y['recall_cum']], color='#9467bd', s=120, marker='v', label='Semantic Stratified Partition (n=10)', zorder=6)
ax4.scatter([jaccard_y['diversity']], [jaccard_y['recall_cum']], color='#ff7f0e', s=120, marker='P', label='Soft-Jaccard DivLoss', zorder=6)

ax4.set_title('Yelp 10-Core: Diversity vs Total Recall (recall_cum)', fontsize=12, fontweight='bold', pad=12)
ax4.set_xlabel('Diversity (1 - Overlap)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Total Recall (10-Trial Cumulative Recall)', fontsize=11, fontweight='bold')
ax4.legend(loc='upper left', fontsize=8.5, frameon=True, facecolor='white', framealpha=0.95)
ax4.set_ylim(0.00, 0.016)

fig2.tight_layout()
fig2.savefig('report/tradeoff_comparison_recall.png', dpi=300, bbox_inches='tight')
plt.close(fig2)

# -------------------------------------------------------------
# 3. MovieLens 1M Dedicated 2x2 Tradeoff Plot
# -------------------------------------------------------------
fig_ml, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

noise_ra_ml = [0.0093, 0.0093, 0.0094, 0.0088]

# (0, 0) Total Slate Precision (%)
ax1.plot(pl_ml['diversity'], pl_ml['slate_precision_pct'], 'o-', color='#1f77b4', linewidth=2.5, markersize=8, label='2-Stage Plackett-Luce (M=200)', zorder=5)
for _, row in pl_ml.iterrows():
    ax1.annotate(f"tau={row['tau']}", (row['diversity'], row['slate_precision_pct']), textcoords="offset points", xytext=(8,-4), ha='left', fontsize=8.5, fontweight='bold', color='#1f77b4')
ax1.plot(noise_div_ml, noise_prec_ml, 's--', color='#d62728', linewidth=2.0, markersize=7, label='Simple Output Noise', zorder=4)
ax1.annotate('sigma=0.05', (noise_div_ml[2], noise_prec_ml[2]), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8.5, color='#d62728')
ax1.scatter([base_ml['diversity']], [base_ml['slate_precision_pct']], color='#7f7f7f', s=120, marker='D', label='Base Two-Tower', zorder=6)
ax1.scatter([part_ml['diversity']], [part_ml['slate_precision_pct']], color='#2ca02c', s=120, marker='^', label='Random Item Partition (n=10)', zorder=6)
ax1.scatter([strat_ml['diversity']], [strat_ml['slate_precision_pct']], color='#9467bd', s=120, marker='v', label='Semantic Stratified Partition (n=10)', zorder=6)
ax1.scatter([jaccard_ml['diversity']], [jaccard_ml['slate_precision_pct']], color='#ff7f0e', s=120, marker='P', label='Soft-Jaccard DivLoss', zorder=6)
ax1.set_title('(a) MovieLens 1M: Diversity vs Total Slate Precision (%)', fontsize=11, fontweight='bold', pad=10)
ax1.set_xlabel('Diversity (1 - Overlap)', fontsize=10, fontweight='bold')
ax1.set_ylabel('Total Slate Precision (%)', fontsize=10, fontweight='bold')
ax1.legend(loc='upper left', fontsize=8, frameon=True, facecolor='white', framealpha=0.95)
ax1.set_ylim(0.50, 1.40)

# (0, 1) Total Recall (Cumulative)
ax2.plot(pl_ml['diversity'], pl_ml['recall_cum'], 'o-', color='#1f77b4', linewidth=2.5, markersize=8, label='2-Stage Plackett-Luce (M=200)', zorder=5)
for _, row in pl_ml.iterrows():
    ax2.annotate(f"tau={row['tau']}", (row['diversity'], row['recall_cum']), textcoords="offset points", xytext=(8,-4), ha='left', fontsize=8.5, fontweight='bold', color='#1f77b4')
ax2.plot(noise_div_ml, noise_recall_ml, 's--', color='#d62728', linewidth=2.0, markersize=7, label='Simple Output Noise', zorder=4)
ax2.annotate('sigma=0.05', (noise_div_ml[2], noise_recall_ml[2]), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8.5, color='#d62728')
ax2.scatter([base_ml['diversity']], [base_ml['recall_cum']], color='#7f7f7f', s=120, marker='D', label='Base Two-Tower', zorder=6)
ax2.scatter([part_ml['diversity']], [part_ml['recall_cum']], color='#2ca02c', s=120, marker='^', label='Random Item Partition (n=10)', zorder=6)
ax2.scatter([strat_ml['diversity']], [strat_ml['recall_cum']], color='#9467bd', s=120, marker='v', label='Semantic Stratified Partition (n=10)', zorder=6)
ax2.scatter([jaccard_ml['diversity']], [jaccard_ml['recall_cum']], color='#ff7f0e', s=120, marker='P', label='Soft-Jaccard DivLoss', zorder=6)
ax2.set_title('(b) MovieLens 1M: Diversity vs Total Recall (recall_cum)', fontsize=11, fontweight='bold', pad=10)
ax2.set_xlabel('Diversity (1 - Overlap)', fontsize=10, fontweight='bold')
ax2.set_ylabel('Total Recall (10-Trial Cumulative Recall)', fontsize=10, fontweight='bold')
ax2.legend(loc='upper left', fontsize=8, frameon=True, facecolor='white', framealpha=0.95)
ax2.set_ylim(0.00, 0.15)

# (1, 0) Average Precision (%)
ax3.plot(pl_ml['diversity'], pl_ml['slate_precision_pct'], 'o-', color='#1f77b4', linewidth=2.5, markersize=8, label='2-Stage Plackett-Luce (M=200)', zorder=5)
for _, row in pl_ml.iterrows():
    ax3.annotate(f"tau={row['tau']}", (row['diversity'], row['slate_precision_pct']), textcoords="offset points", xytext=(8,-4), ha='left', fontsize=8.5, fontweight='bold', color='#1f77b4')
ax3.plot(noise_div_ml, noise_prec_ml, 's--', color='#d62728', linewidth=2.0, markersize=7, label='Simple Output Noise', zorder=4)
ax3.annotate('sigma=0.05', (noise_div_ml[2], noise_prec_ml[2]), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8.5, color='#d62728')
ax3.scatter([base_ml['diversity']], [base_ml['slate_precision_pct']], color='#7f7f7f', s=120, marker='D', label='Base Two-Tower', zorder=6)
ax3.scatter([part_ml['diversity']], [part_ml['slate_precision_pct']], color='#2ca02c', s=120, marker='^', label='Random Item Partition (n=10)', zorder=6)
ax3.scatter([strat_ml['diversity']], [strat_ml['slate_precision_pct']], color='#9467bd', s=120, marker='v', label='Semantic Stratified Partition (n=10)', zorder=6)
ax3.scatter([jaccard_ml['diversity']], [jaccard_ml['slate_precision_pct']], color='#ff7f0e', s=120, marker='P', label='Soft-Jaccard DivLoss', zorder=6)
ax3.set_title('(c) MovieLens 1M: Diversity vs Average Precision (%)', fontsize=11, fontweight='bold', pad=10)
ax3.set_xlabel('Diversity (1 - Overlap)', fontsize=10, fontweight='bold')
ax3.set_ylabel('Average Precision (Per-Trial %)', fontsize=10, fontweight='bold')
ax3.legend(loc='upper left', fontsize=8, frameon=True, facecolor='white', framealpha=0.95)
ax3.set_ylim(0.50, 1.40)

# (1, 1) Average Recall (Per-Trial)
ax4.plot(pl_ml['diversity'], pl_ml['recall_avg'], 'o-', color='#1f77b4', linewidth=2.5, markersize=8, label='2-Stage Plackett-Luce (M=200)', zorder=5)
for _, row in pl_ml.iterrows():
    ax4.annotate(f"tau={row['tau']}", (row['diversity'], row['recall_avg']), textcoords="offset points", xytext=(8,-4), ha='left', fontsize=8.5, fontweight='bold', color='#1f77b4')
ax4.plot(noise_div_ml, noise_ra_ml, 's--', color='#d62728', linewidth=2.0, markersize=7, label='Simple Output Noise', zorder=4)
ax4.annotate('sigma=0.05', (noise_div_ml[2], noise_ra_ml[2]), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8.5, color='#d62728')
ax4.scatter([base_ml['diversity']], [base_ml['recall_avg']], color='#7f7f7f', s=120, marker='D', label='Base Two-Tower', zorder=6)
ax4.scatter([part_ml['diversity']], [part_ml['recall_avg']], color='#2ca02c', s=120, marker='^', label='Random Item Partition (n=10)', zorder=6)
ax4.scatter([strat_ml['diversity']], [strat_ml['recall_avg']], color='#9467bd', s=120, marker='v', label='Semantic Stratified Partition (n=10)', zorder=6)
ax4.scatter([jaccard_ml['diversity']], [jaccard_ml['recall_avg']], color='#ff7f0e', s=120, marker='P', label='Soft-Jaccard DivLoss', zorder=6)
ax4.set_title('(d) MovieLens 1M: Diversity vs Average Recall (recall_avg)', fontsize=11, fontweight='bold', pad=10)
ax4.set_xlabel('Diversity (1 - Overlap)', fontsize=10, fontweight='bold')
ax4.set_ylabel('Average Recall (Per-Trial Recall@10)', fontsize=10, fontweight='bold')
ax4.legend(loc='upper left', fontsize=8, frameon=True, facecolor='white', framealpha=0.95)
ax4.set_ylim(0.006, 0.018)

fig_ml.tight_layout()
fig_ml.savefig('report/tradeoff_movielens.png', dpi=300, bbox_inches='tight')
plt.close(fig_ml)

# -------------------------------------------------------------
# 4. Yelp 10-Core Dedicated 2x2 Tradeoff Plot
# -------------------------------------------------------------
fig_yelp, ((ax_y1, ax_y2), (ax_y3, ax_y4)) = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

noise_ra_y = [0.0010, 0.0007]

# (0, 0) Total Slate Precision (%)
ax_y1.plot(pl_y['diversity'], pl_y['slate_precision_pct'], 'o-', color='#1f77b4', linewidth=2.5, markersize=8, label='2-Stage Plackett-Luce (M=200)', zorder=5)
for _, row in pl_y.iterrows():
    ax_y1.annotate(f"tau={row['tau']}", (row['diversity'], row['slate_precision_pct']), textcoords="offset points", xytext=(8,-4), ha='left', fontsize=8.5, fontweight='bold', color='#1f77b4')
ax_y1.plot(noise_div_y, noise_prec_y, 's--', color='#d62728', linewidth=2.0, markersize=7, label='Simple Output Noise', zorder=4)
ax_y1.annotate('sigma=0.05', (noise_div_y[1], noise_prec_y[1]), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8.5, color='#d62728')
ax_y1.scatter([base_y['diversity']], [base_y['slate_precision_pct']], color='#7f7f7f', s=120, marker='D', label='Base Two-Tower', zorder=6)
ax_y1.scatter([part_y['diversity']], [part_y['slate_precision_pct']], color='#2ca02c', s=120, marker='^', label='Random Item Partition (n=10)', zorder=6)
ax_y1.scatter([strat_y['diversity']], [strat_y['slate_precision_pct']], color='#9467bd', s=120, marker='v', label='Semantic Stratified Partition (n=10)', zorder=6)
ax_y1.scatter([jaccard_y['diversity']], [jaccard_y['slate_precision_pct']], color='#ff7f0e', s=120, marker='P', label='Soft-Jaccard DivLoss', zorder=6)
ax_y1.set_title('(a) Yelp 10-Core: Diversity vs Total Slate Precision (%)', fontsize=11, fontweight='bold', pad=10)
ax_y1.set_xlabel('Diversity (1 - Overlap)', fontsize=10, fontweight='bold')
ax_y1.set_ylabel('Total Slate Precision (%)', fontsize=10, fontweight='bold')
ax_y1.legend(loc='upper left', fontsize=8, frameon=True, facecolor='white', framealpha=0.95)
ax_y1.set_ylim(0.010, 0.048)

# (0, 1) Total Recall (Cumulative)
ax_y2.plot(pl_y['diversity'], pl_y['recall_cum'], 'o-', color='#1f77b4', linewidth=2.5, markersize=8, label='2-Stage Plackett-Luce (M=200)', zorder=5)
for _, row in pl_y.iterrows():
    ax_y2.annotate(f"tau={row['tau']}", (row['diversity'], row['recall_cum']), textcoords="offset points", xytext=(8,-4), ha='left', fontsize=8.5, fontweight='bold', color='#1f77b4')
ax_y2.plot(noise_div_y, noise_recall_y, 's--', color='#d62728', linewidth=2.0, markersize=7, label='Simple Output Noise', zorder=4)
ax_y2.annotate('sigma=0.05', (noise_div_y[1], noise_recall_y[1]), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8.5, color='#d62728')
ax_y2.scatter([base_y['diversity']], [base_y['recall_cum']], color='#7f7f7f', s=120, marker='D', label='Base Two-Tower', zorder=6)
ax_y2.scatter([part_y['diversity']], [part_y['recall_cum']], color='#2ca02c', s=120, marker='^', label='Random Item Partition (n=10)', zorder=6)
ax_y2.scatter([strat_y['diversity']], [strat_y['recall_cum']], color='#9467bd', s=120, marker='v', label='Semantic Stratified Partition (n=10)', zorder=6)
ax_y2.scatter([jaccard_y['diversity']], [jaccard_y['recall_cum']], color='#ff7f0e', s=120, marker='P', label='Soft-Jaccard DivLoss', zorder=6)
ax_y2.set_title('(b) Yelp 10-Core: Diversity vs Total Recall (recall_cum)', fontsize=11, fontweight='bold', pad=10)
ax_y2.set_xlabel('Diversity (1 - Overlap)', fontsize=10, fontweight='bold')
ax_y2.set_ylabel('Total Recall (10-Trial Cumulative Recall)', fontsize=10, fontweight='bold')
ax_y2.legend(loc='upper left', fontsize=8, frameon=True, facecolor='white', framealpha=0.95)
ax_y2.set_ylim(0.00, 0.016)

# (1, 0) Average Precision (%)
ax_y3.plot(pl_y['diversity'], pl_y['slate_precision_pct'], 'o-', color='#1f77b4', linewidth=2.5, markersize=8, label='2-Stage Plackett-Luce (M=200)', zorder=5)
for _, row in pl_y.iterrows():
    ax_y3.annotate(f"tau={row['tau']}", (row['diversity'], row['slate_precision_pct']), textcoords="offset points", xytext=(8,-4), ha='left', fontsize=8.5, fontweight='bold', color='#1f77b4')
ax_y3.plot(noise_div_y, noise_prec_y, 's--', color='#d62728', linewidth=2.0, markersize=7, label='Simple Output Noise', zorder=4)
ax_y3.annotate('sigma=0.05', (noise_div_y[1], noise_prec_y[1]), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8.5, color='#d62728')
ax_y3.scatter([base_y['diversity']], [base_y['slate_precision_pct']], color='#7f7f7f', s=120, marker='D', label='Base Two-Tower', zorder=6)
ax_y3.scatter([part_y['diversity']], [part_y['slate_precision_pct']], color='#2ca02c', s=120, marker='^', label='Random Item Partition (n=10)', zorder=6)
ax_y3.scatter([strat_y['diversity']], [strat_y['slate_precision_pct']], color='#9467bd', s=120, marker='v', label='Semantic Stratified Partition (n=10)', zorder=6)
ax_y3.scatter([jaccard_y['diversity']], [jaccard_y['slate_precision_pct']], color='#ff7f0e', s=120, marker='P', label='Soft-Jaccard DivLoss', zorder=6)
ax_y3.set_title('(c) Yelp 10-Core: Diversity vs Average Precision (%)', fontsize=11, fontweight='bold', pad=10)
ax_y3.set_xlabel('Diversity (1 - Overlap)', fontsize=10, fontweight='bold')
ax_y3.set_ylabel('Average Precision (Per-Trial %)', fontsize=10, fontweight='bold')
ax_y3.legend(loc='upper left', fontsize=8, frameon=True, facecolor='white', framealpha=0.95)
ax_y3.set_ylim(0.010, 0.048)

# (1, 1) Average Recall (Per-Trial)
ax_y4.plot(pl_y['diversity'], pl_y['recall_avg'], 'o-', color='#1f77b4', linewidth=2.5, markersize=8, label='2-Stage Plackett-Luce (M=200)', zorder=5)
for _, row in pl_y.iterrows():
    ax_y4.annotate(f"tau={row['tau']}", (row['diversity'], row['recall_avg']), textcoords="offset points", xytext=(8,-4), ha='left', fontsize=8.5, fontweight='bold', color='#1f77b4')
ax_y4.plot(noise_div_y, noise_ra_y, 's--', color='#d62728', linewidth=2.0, markersize=7, label='Simple Output Noise', zorder=4)
ax_y4.annotate('sigma=0.05', (noise_div_y[1], noise_ra_y[1]), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8.5, color='#d62728')
ax_y4.scatter([base_y['diversity']], [base_y['recall_avg']], color='#7f7f7f', s=120, marker='D', label='Base Two-Tower', zorder=6)
ax_y4.scatter([part_y['diversity']], [part_y['recall_avg']], color='#2ca02c', s=120, marker='^', label='Random Item Partition (n=10)', zorder=6)
ax_y4.scatter([strat_y['diversity']], [strat_y['recall_avg']], color='#9467bd', s=120, marker='v', label='Semantic Stratified Partition (n=10)', zorder=6)
ax_y4.scatter([jaccard_y['diversity']], [jaccard_y['recall_avg']], color='#ff7f0e', s=120, marker='P', label='Soft-Jaccard DivLoss', zorder=6)
ax_y4.set_title('(d) Yelp 10-Core: Diversity vs Average Recall (recall_avg)', fontsize=11, fontweight='bold', pad=10)
ax_y4.set_xlabel('Diversity (1 - Overlap)', fontsize=10, fontweight='bold')
ax_y4.set_ylabel('Average Recall (Per-Trial Recall@10)', fontsize=10, fontweight='bold')
ax_y4.legend(loc='upper left', fontsize=8, frameon=True, facecolor='white', framealpha=0.95)
ax_y4.set_ylim(0.0004, 0.0022)

fig_yelp.tight_layout()
fig_yelp.savefig('report/tradeoff_yelp.png', dpi=300, bbox_inches='tight')
plt.close(fig_yelp)

print("Precision plot saved to report/tradeoff_comparison_precision.png")
print("Recall plot saved to report/tradeoff_comparison_recall.png")
print("MovieLens 2x2 plot saved to report/tradeoff_movielens.png")
print("Yelp 2x2 plot saved to report/tradeoff_yelp.png")


