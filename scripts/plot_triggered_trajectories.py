"""Plot comprehensive comparison figures for Triggered Physiological Dynamics Forecasters:
fig9_triggered_forecasting_cases.png:
- Detailed trajectory plots on representative real clinical cases (rising postprandial surges & falling hypoglycemic crashes)
- Comparison of Ground Truth, Triggered Dynamics Model, Persistence, Linear Momentum, and LSTM Forecaster
- Quantitative bar charts of RMSE and Extrema (Peak/Nadir) Prediction MAE
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.titlesize'] = 15

os.makedirs('reports/figures', exist_ok=True)

# 1. Load data
cases_file = 'runs/triggered_forecasting_cases.json'
comp_file = 'runs/triggered_forecasting_comparison.csv'
risk_file = 'runs/triggered_risk_alerts.json'

if not os.path.exists(cases_file) or not os.path.exists(comp_file):
    print("Required run files not found yet. Exiting.")
    exit(1)

with open(cases_file, 'r') as f:
    cases = json.load(f)
df_comp = pd.read_csv(comp_file)

rising_cases = cases.get('rising', [])
falling_cases = cases.get('falling', [])

# Find best representative cases
# Case 1: Steep rising surge
best_rise_1 = None
max_delta_rise = 0
for c in rising_cases:
    delta = max(c['future_y']) - c['g_t']
    if 45.0 < delta < 90.0 and max(c['future_y']) > 180.0:
        best_rise_1 = c
        break
if best_rise_1 is None and len(rising_cases) > 0:
    best_rise_1 = rising_cases[0]

# Case 2: Moderate rising surge
best_rise_2 = None
for c in rising_cases:
    delta = max(c['future_y']) - c['g_t']
    if 30.0 < delta < 55.0 and c != best_rise_1:
        best_rise_2 = c
        break
if best_rise_2 is None and len(rising_cases) > 1:
    best_rise_2 = rising_cases[1]

# Case 3: Hypoglycemic crash
best_fall_1 = None
for c in falling_cases:
    delta = min(c['future_y']) - c['g_t']
    if delta < -35.0 and min(c['future_y']) < 75.0:
        best_fall_1 = c
        break
if best_fall_1 is None and len(falling_cases) > 0:
    best_fall_1 = falling_cases[0]

# Case 4: Deep plunge
best_fall_2 = None
for c in falling_cases:
    delta = min(c['future_y']) - c['g_t']
    if delta < -45.0 and c != best_fall_1:
        best_fall_2 = c
        break
if best_fall_2 is None and len(falling_cases) > 1:
    best_fall_2 = falling_cases[1]

# Create Figure (2 rows, 3 columns)
fig, axes = plt.subplots(2, 3, figsize=(18, 11))

def plot_case(ax, case_data, title, mode="rising"):
    hist = np.array(case_data['history'])
    fut_true = np.array(case_data['future_y'])
    fut_pers = np.array(case_data['pred_pers'])
    fut_lin = np.array(case_data['pred_lin'])
    fut_lstm = np.array(case_data['pred_lstm'])
    fut_trig = np.array(case_data['pred_trig'])
    
    t_hist = np.arange(-12 * 5, 0, 5) # -60 to -5 min
    t_fut = np.arange(0, 24 * 5, 5)   # 0 to 115 min
    
    # Connect t=0
    t_hist_conn = np.append(t_hist, 0)
    v_hist_conn = np.append(hist, case_data['g_t'])
    
    # Plot History
    ax.plot(t_hist_conn, v_hist_conn, color='#2c3e50', linewidth=2.5, marker='o', markersize=3, label='CGM History (Past 60m)')
    
    # Plot Clinical Thresholds
    ax.axhline(180, color='#e74c3c', linestyle=':', alpha=0.6, label='Hyper Target (>180 mg/dL)' if mode=='rising' else "")
    ax.axhline(70, color='#c0392b', linestyle='--', alpha=0.6, label='Hypo Alert (<70 mg/dL)' if mode=='falling' else "")
    ax.axvline(0, color='gray', linestyle='-', alpha=0.5)
    
    # Plot Future True
    ax.plot(t_fut, fut_true, color='#000000', linewidth=3.0, label='Ground Truth Future', zorder=5)
    
    # Plot Models
    ax.plot(t_fut, fut_trig, color='#e67e22', linewidth=2.5, linestyle='-', marker='s', markersize=3, label='Triggered Dynamics (Ours)', zorder=4)
    ax.plot(t_fut, fut_pers, color='#7f8c8d', linewidth=2.0, linestyle='--', label='Persistence (Flat)', zorder=2)
    ax.plot(t_fut, fut_lstm, color='#2980b9', linewidth=2.0, linestyle='-.', label='LSTM (Unconstrained)', zorder=3)
    ax.plot(t_fut, fut_lin, color='#27ae60', linewidth=1.5, linestyle=':', label='Linear Momentum', zorder=1)
    
    # Highlight Trigger Point
    ax.scatter(0, case_data['g_t'], color='#e74c3c' if mode=='rising' else '#2980b9', s=100, zorder=6,
               edgecolor='black', linewidth=1.2, label=f"Trigger (ROC={case_data['roc15']:.2f})")
    
    ax.set_title(title, fontweight='bold', fontsize=11)
    ax.set_xlabel('Time Relative to Trigger (minutes)', fontweight='bold')
    ax.set_ylabel('Glucose Value (mg/dL)', fontweight='bold')
    ax.set_xlim(-65, 120)
    ax.legend(loc='best', fontsize=8.5, frameon=True, facecolor='white', framealpha=0.9)

# Panel (0,0): Rising Case 1
plot_case(axes[0, 0], best_rise_1, '(A) Postprandial Surge: Hyperglycemia Peak Breach', mode='rising')

# Panel (0,1): Rising Case 2
plot_case(axes[0, 1], best_rise_2, '(B) Postprandial Surge: Moderate Absorption Curve', mode='rising')

# Panel (1,0): Falling Case 1
plot_case(axes[1, 0], best_fall_1, '(C) Rapid Fall: Critical Hypoglycemia Alert', mode='falling')

# Panel (1,1): Falling Case 2
plot_case(axes[1, 1], best_fall_2, '(D) Rapid Fall: Steep Glycemic Plunge', mode='falling')

# Panel (0,2): Trajectory RMSE Bar Chart
ax_rmse = axes[0, 2]
models = ['Persistence', 'Linear_Momentum', 'LSTM_Unconstrained', 'Triggered_Dynamics']
model_labels = ['Persistence\n(Flat line)', 'Linear\nMomentum', 'LSTM\n(Global)', 'Triggered\nDynamics']
colors = ['#7f8c8d', '#27ae60', '#2980b9', '#e67e22']

# Let's extract 60m RMSE for Rising and Falling
rise_rows = df_comp[df_comp['mode'] == 'rising']
fall_rows = df_comp[df_comp['mode'] == 'falling']

x_pos = np.arange(len(models))
w = 0.35

rmse_rise_60 = [rise_rows[rise_rows['model'] == m]['60m_rmse'].values[0] for m in models]
rmse_fall_60 = [fall_rows[fall_rows['model'] == m]['60m_rmse'].values[0] for m in models]

rects1 = ax_rmse.bar(x_pos - w/2, rmse_rise_60, w, label='Rising Events (1h)', color='#e67e22', alpha=0.85, edgecolor='black')
rects2 = ax_rmse.bar(x_pos + w/2, rmse_fall_60, w, label='Falling Events (1h)', color='#3498db', alpha=0.85, edgecolor='black')

for rects in [rects1, rects2]:
    for rect in rects:
        h = rect.get_height()
        ax_rmse.annotate(f'{h:.1f}',
                         xy=(rect.get_x() + rect.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points",
                         ha='center', va='bottom', fontsize=8.5, fontweight='bold')

ax_rmse.set_xticks(x_pos)
ax_rmse.set_xticklabels(model_labels, fontsize=9.5, fontweight='bold')
ax_rmse.set_ylabel('60-min Forecast RMSE (mg/dL) [Lower is Better]', fontweight='bold')
ax_rmse.set_title('(E) Trajectory RMSE on Triggered Events', fontweight='bold')
ax_rmse.legend(frameon=True, facecolor='white', loc='upper right')
ax_rmse.set_ylim(0, max(max(rmse_rise_60), max(rmse_fall_60)) * 1.25)

# Panel (1,2): Extrema Error MAE
ax_ext = axes[1, 2]
ext_rise = [rise_rows[rise_rows['model'] == m]['extrema_mae'].values[0] for m in models]
ext_fall = [fall_rows[fall_rows['model'] == m]['extrema_mae'].values[0] for m in models]

rects3 = ax_ext.bar(x_pos - w/2, ext_rise, w, label='Peak MAE (Rising)', color='#d35400', alpha=0.85, edgecolor='black')
rects4 = ax_ext.bar(x_pos + w/2, ext_fall, w, label='Nadir MAE (Falling)', color='#2980b9', alpha=0.85, edgecolor='black')

for rects in [rects3, rects4]:
    for rect in rects:
        h = rect.get_height()
        ax_ext.annotate(f'{h:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8.5, fontweight='bold')

ax_ext.set_xticks(x_pos)
ax_ext.set_xticklabels(model_labels, fontsize=9.5, fontweight='bold')
ax_ext.set_ylabel('Extrema Absolute Error (mg/dL) [Lower is Better]', fontweight='bold')
ax_ext.set_title('(F) Peak & Nadir Extrema Prediction Error', fontweight='bold')
ax_ext.legend(frameon=True, facecolor='white', loc='upper right')
ax_ext.set_ylim(0, max(max(ext_rise), max(ext_fall)) * 1.25)

plt.tight_layout()
out_fig = 'reports/figures/fig9_triggered_forecasting_cases.png'
plt.savefig(out_fig, dpi=300, bbox_inches='tight')
print(f"Successfully saved {out_fig}")
