"""Plot comprehensive benchmark figures for Amazon Chronos vs. TimesFM, LSTM/GRU, and Persistence:
1. reports/figures/fig10_chronos_multihorizon_benchmark.png: Multi-horizon RMSE and delta reduction vs Persistence across 6 models
2. reports/figures/fig11_chronos_probabilistic_trajectories.png: 4-panel case study with Chronos 80% prediction intervals (p10-p90) on clinical events
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['figure.titlesize'] = 15

os.makedirs('reports/figures', exist_ok=True)

# -------------------------------------------------------------
# Figure 10: Multi-Horizon Benchmark Comparison (RMSE & Delta %)
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6.5))

df_fc = pd.read_csv('runs/gpu_forecasting_comparison.csv')

models_order = [
    'Persistence Baseline',
    'Chronos-Bolt-Tiny-9M (Zero-Shot)',
    'Chronos-Bolt-Base-200M (Zero-Shot)',
    'TimesFM-2.5-200M (Zero-Shot)',
    'LSTM Forecaster (Dedicated GPU)',
    'GRU Forecaster (Dedicated GPU)'
]

palette = ['#7f8c8d', '#af7ac5', '#7d3c98', '#2980b9', '#e67e22', '#27ae60']

x = np.arange(3)
width = 0.14

for idx, (model, color) in enumerate(zip(models_order, palette)):
    m_data = df_fc[df_fc['model'] == model].sort_values('steps')
    rmses = m_data['rmse'].values
    offset = (idx - 2.5) * width
    rects = ax1.bar(x + offset, rmses, width, label=model, color=color, alpha=0.92, edgecolor='black', linewidth=0.7)
    for rect in rects:
        h = rect.get_height()
        ax1.annotate(f'{h:.1f}',
                     xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=8.5, fontweight='bold')

ax1.set_xticks(x)
ax1.set_xticklabels(['30 min (6 steps)', '60 min (12 steps)', '120 min / 2 hours (24 steps)'], fontweight='bold')
ax1.set_ylabel('RMSE (mg/dL) [Lower is Better]', fontweight='bold')
ax1.set_title('(A) Multi-Horizon Forecast RMSE: Foundation vs Dedicated Models', fontweight='bold')
ax1.legend(frameon=True, facecolor='white', loc='upper left', fontsize=9.5)
ax1.set_ylim(0, 32)

# Subplot 2: Relative Error Reduction vs Persistence (%)
pers_rmses = df_fc[df_fc['model'] == 'Persistence Baseline'].sort_values('steps')['rmse'].values

models_comp = models_order[1:] # omit persistence
palette_comp = palette[1:]

for idx, (model, color) in enumerate(zip(models_comp, palette_comp)):
    m_data = df_fc[df_fc['model'] == model].sort_values('steps')
    rmses = m_data['rmse'].values
    deltas = ((pers_rmses - rmses) / pers_rmses) * 100.0
    offset = (idx - 2) * (width * 1.1)
    rects = ax2.bar(x + offset, deltas, width * 1.1, label=model, color=color, alpha=0.92, edgecolor='black', linewidth=0.7)
    for rect in rects:
        h = rect.get_height()
        ax2.annotate(f'{h:+.1f}%',
                     xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=8.5, fontweight='bold')

ax2.set_xticks(x)
ax2.set_xticklabels(['30 min (6 steps)', '60 min (12 steps)', '120 min / 2 hours (24 steps)'], fontweight='bold')
ax2.set_ylabel('Error Reduction vs Persistence (%) [Higher is Better]', fontweight='bold')
ax2.set_title('(B) Relative Accuracy Gain over Persistence Baseline (Δ%)', fontweight='bold')
ax2.axhline(0, color='gray', linestyle='--', linewidth=1.0)
ax2.legend(frameon=True, facecolor='white', loc='upper right', fontsize=9.5)
ax2.set_ylim(-1, 16)

plt.tight_layout()
fig10_path = 'reports/figures/fig10_chronos_multihorizon_benchmark.png'
plt.savefig(fig10_path, dpi=300)
plt.close()
print(f"[Plot] Saved Figure 10 to {fig10_path}")


# -------------------------------------------------------------
# Figure 11: Probabilistic Trajectories & Uncertainty Bands
# -------------------------------------------------------------
with open('runs/chronos_forecast_trajectories.json', 'r') as f:
    trajectories = json.load(f)

# Pick 4 representative trajectories
cases = [
    (0, "(A) Rapid Postprandial Glucose Surge (Meal Excursion)"),
    (1, "(B) Sharp Glycemic Decline & Hypoglycemia Risk (< 70 mg/dL)"),
    (2, "(C) Dynamic Glycemic Inflection & Turning Point"),
    (7, "(D) Nocturnal Euglycemic Steady-State (Tight Interval)"),
]

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

time_hist = np.arange(-24, 0) * 5  # minutes relative to forecast start
time_pred = np.arange(1, 25) * 5   # minutes future

for ax, (case_idx, title) in zip(axes.flatten(), cases):
    data = trajectories[case_idx]
    hist = data["history_2h"]
    target = data["target"]
    pers = data["persistence_pred"]
    chr_base = data["chronos_base_mean"]
    p10 = data["chronos_base_p10"]
    p90 = data["chronos_base_p90"]

    # Target glycemic range (70-180 mg/dL)
    ax.axhspan(70, 180, color='#2ecc71', alpha=0.10, label='Target Range (70-180 mg/dL)')
    # Hypoglycemia danger zone (< 70 mg/dL)
    ax.axhspan(30, 70, color='#e74c3c', alpha=0.12, label='Hypoglycemia Zone (< 70 mg/dL)')
    ax.axhline(70, color='#e74c3c', linestyle=':', linewidth=1.5, alpha=0.8)

    # Historical line
    ax.plot(time_hist, hist, color='#2c3e50', linewidth=2.4, marker='o', markersize=3.5, label='CGM History (-2h to 0)')
    
    # Ground truth future
    ax.plot(time_pred, target, color='#27ae60', linewidth=2.6, marker='s', markersize=3.5, label='Ground Truth Future (+2h)')
    
    # Persistence baseline
    ax.plot(time_pred, pers, color='#7f8c8d', linestyle='--', linewidth=2.0, label='Persistence Baseline')
    
    # Chronos-Bolt-Base point forecast
    ax.plot(time_pred, chr_base, color='#8e44ad', linewidth=2.5, marker='^', markersize=3.5, label='Chronos-Bolt-Base (Mean)')
    
    # Chronos 80% prediction interval (p10 to p90)
    ax.fill_between(time_pred, p10, p90, color='#9b59b6', alpha=0.25, label='Chronos 80% Conf. Interval [p10-p90]')
    
    # Mark forecast origin (t=0)
    ax.axvline(0, color='black', linestyle='-', linewidth=1.2, alpha=0.7)
    
    ax.set_xlabel('Time Relative to Prediction Origin (minutes)', fontweight='bold')
    ax.set_ylabel('Glucose (mg/dL)', fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=13)
    ax.set_xlim(-125, 125)
    ax.set_xticks([-120, -90, -60, -30, 0, 30, 60, 90, 120])
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Place legend cleanly
    if case_idx == 0:
        ax.legend(frameon=True, facecolor='white', loc='upper left', fontsize=9.2, ncol=2)

plt.tight_layout()
fig11_path = 'reports/figures/fig11_chronos_probabilistic_trajectories.png'
plt.savefig(fig11_path, dpi=300)
plt.close()
print(f"[Plot] Saved Figure 11 to {fig11_path}")
