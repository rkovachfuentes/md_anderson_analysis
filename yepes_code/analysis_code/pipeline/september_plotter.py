import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# 1. Load and clean data (Include 'max_V' in dropna)
df = pd.read_csv("0922_generator_out_updated.csv")
hv = 100.0

target_resistances = [100]  

df_clean = df[df['beam'] != 'Noise/Rejected'].dropna(subset=['dose', 'area', 'max_V']).copy()
df_clean = df_clean[df_clean['resistance'].isin(target_resistances)].copy()

# 2. Mappings for markers (beam) and colors (resistance)
beam_list = sorted(df_clean['beam'].unique(), key=lambda x: float(x))
markers = ['o', 's', '^', 'D', 'v', 'P', '*']
beam_marker_map = {b: markers[i % len(markers)] for i, b in enumerate(beam_list)}

res_list = sorted(df_clean['resistance'].unique())
res_color_map = {100: '#1f77b4', 10: '#ff7f0e'}  

collimators = sorted(df_clean['collimator'].unique())

# 3. Global shared legend setup
legend_elements = [
    Line2D([0], [0], color='none', label='-- Beam (Shape) --')
]
for b in beam_list:
    legend_elements.append(Line2D([0], [0], marker=beam_marker_map[b], color='w', 
                                  markerfacecolor='gray', markeredgecolor='k', markeredgewidth=0.5,
                                  markersize=8, label=f'Beam {b}'))

legend_elements.append(Line2D([0], [0], color='none', label=''))  # Spacer
legend_elements.append(Line2D([0], [0], color='none', label='-- Resistance (Color) --'))
for r in res_list:
    legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor=res_color_map[r], markeredgecolor='k', markeredgewidth=0.5,
                                  markersize=8, label=f'{r} Ω'))


# 4. Parametrized Helper Function (Accepts y_var and y_label dynamically)
def draw_collimator_plot(ax, sub, y_var, y_label):
    for (b, r), group in sub.groupby(['beam', 'resistance']):
        ax.scatter(group['dose'], group[y_var], 
                   marker=beam_marker_map[b], 
                   color=res_color_map[r], 
                   alpha=0.85, edgecolors='k', linewidths=0.5, s=60,
                   zorder=3 if r == 10 else 2)
    
    # Linear fit
    slope, intercept = np.polyfit(sub['dose'], sub[y_var], 1)
    y_pred = slope * sub['dose'] + intercept
    r_squared = 1 - (np.sum((sub[y_var] - y_pred)**2) / np.sum((sub[y_var] - np.mean(sub[y_var]))**2))
    
    x_line = np.linspace(sub['dose'].min(), sub['dose'].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, linestyle='--', color='#333333', zorder=1,
            label=f'Fit: y = {slope:.2e}x + {intercept:.1f}\n($R^2 = {r_squared:.3f}$)')
    
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper left')


# =========================================================
# LOOP OVER METRICS (Generates both Area and Max V plots)
# =========================================================
metrics = [
    ('area', 'Area', 'area'),
    ('max_V', 'Max V (V)', 'max_v')
]

for y_var, y_label, metric_name in metrics:

    # -----------------------------------------------------
    # OPTION A: Combined Canvas (All Collimators)
    # -----------------------------------------------------
    fig_combined, axes = plt.subplots(1, len(collimators), figsize=(14, 6), sharey=True, layout='constrained')

    for ax, coll in zip(axes, collimators):
        sub = df_clean[df_clean['collimator'] == coll]
        draw_collimator_plot(ax, sub, y_var, y_label)
        ax.set_title(f'Collimator {coll} cm')

    axes[0].set_xlabel('Dose (nC)')
    fig_combined.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.0, 0.5), borderaxespad=0.)
    fig_combined.suptitle(f'Dose vs {y_label} by Collimator, HV = {hv}', fontsize=14)

    plt.savefig(f'dose_vs_{metric_name}_combined_{hv}.png', bbox_inches='tight')
    plt.show()

    # -----------------------------------------------------
    # OPTION B: Separate Canvas per Collimator
    # -----------------------------------------------------
    for coll in collimators:
        fig_single, ax_single = plt.subplots(figsize=(7, 6), layout='constrained')
        sub = df_clean[df_clean['collimator'] == coll]
        
        draw_collimator_plot(ax_single, sub, y_var, y_label)
        
        ax_single.set_xlabel('Dose (nC)')
        ax_single.set_title(f'Collimator {coll} cm')
        fig_single.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.0, 0.5), borderaxespad=0.)
        fig_single.suptitle(f'Dose vs {y_label} - Collimator {coll} cm, HV = {hv}', fontsize=14)
        
        plt.savefig(f'dose_vs_{metric_name}_collimator_{coll}_{hv}.png', bbox_inches='tight')
        plt.show()