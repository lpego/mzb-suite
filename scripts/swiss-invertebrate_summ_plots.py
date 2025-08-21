import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

# Load the CSV
df = pd.read_csv('..\\results\\swiss-invertebrates\\merged_output.csv')

# For plotting, exclude 'errors' pred_class only in plot data, not in the dataframe
if 'pred_class' in df.columns:
    all_pred_classes = [pc for pc in df['pred_class'].unique() if pc != 'errors' and any(df['pred_class'] == pc)]

# Custom split to handle 'mixed_difficulty' in site_sorting
new_cols = ['site_number', 'site_treatment', 'site_sorting', 'site_image', 'site_clip']

def custom_split(row):
    parts = row.split('_')
    if 'mixed_difficulty' in row:
        # Assume 'mixed_difficulty' is always the third part
        idx = parts.index('mixed')
        site_sorting = '_'.join(parts[idx:idx+2])
        # Rebuild the list with the merged 'mixed_difficulty'
        new_parts = parts[:idx] + [site_sorting] + parts[idx+2:]
        # Pad or trim to 5 columns
        return (new_parts + [None]*5)[:5]
    else:
        return (parts + [None]*5)[:5]

split_cols = df['file_noext'].apply(custom_split).apply(pd.Series)
split_cols.columns = new_cols

# Insert new columns immediately after 'file_noext'
file_noext_idx = df.columns.get_loc('file_noext')
for i, col in enumerate(split_cols.columns):
    df.insert(file_noext_idx + 1 + i, col, split_cols[col])

# Save the updated DataFrame
df.to_csv('..\\results\\swiss-invertebrates\\merged_output_with_split.csv', index=False)

# Ensure the columns are present
required_cols = ['site_number', 'site_treatment', 'skel_length_mm', 'nn_pred_body']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame.")

# Create output directory for plots
output_dir = '..\\results\\swiss-invertebrates\\site_boxplots'
os.makedirs(output_dir, exist_ok=True)

# Use a pastel color palette for treatments
pastel_colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
pastel_palette = [c for c in pastel_colors if 'light' in c or 'pale' in c or 'lavender' in c or 'misty' in c or 'beige' in c or 'peach' in c or 'powder' in c or 'mint' in c or 'honeydew' in c or 'azure' in c or 'seashell' in c or 'lemonchiffon' in c or 'aliceblue' in c or 'linen' in c or 'oldlace' in c or 'blanchedalmond' in c or 'cornsilk' in c or 'ivory' in c or 'ghostwhite' in c or 'floralwhite' in c or 'whitesmoke']

# Define the desired treatment order and their display names
ordered_treatments = [
    'ob',  # Outside Benthos
    'b1', # Benthos 1
    'b2', # Benthos 2
    'SPACE',
    'bd', # Base Drift
    'ur', # Up-Ramping
    'hf1', # High Flow 1
    'hf2'  # High Flow 2
]
treatment_display_names = {
    'ob': 'Outside Benthos',
    'b1': 'Benthos 1',
    'b2': 'Benthos 2',
    'SPACE': 'SPACE',
    'bd': 'Base Drift',
    'ur': 'Up-Ramping',
    'hf1': 'High Flow 1',
    'hf2': 'High Flow 2'
}

# Filter only treatments present in the data and in the desired order
all_treatments = [t for t in ordered_treatments if t in pd.unique(df['site_treatment'])]
# If not enough pastel colors, repeat
while len(pastel_palette) < len(all_treatments):
    pastel_palette = pastel_palette * 2
pastel_palette = pastel_palette[:len(all_treatments)]
treatment_color_map = {t: pastel_palette[i] for i, t in enumerate(all_treatments)}

# Prepare data for plotting
site_numbers = df['site_number'].unique()
n_sites = len(site_numbers)
fig1, axes1 = plt.subplots(n_sites, 2, figsize=(14, 5 * n_sites), squeeze=False)

for idx, site in enumerate(site_numbers):
    site_df = df[df['site_number'] == site]
    # Use the ordered treatments present in this site
    treatments = [t for t in all_treatments if t in site_df['site_treatment'].unique()]
    # Boxplot for skel_length_mm
    data1 = [site_df[site_df['site_treatment'] == t]['skel_length_mm'].dropna() for t in treatments]
    bplot1 = axes1[idx, 0].boxplot(data1, patch_artist=True, labels=[treatment_display_names[t] for t in treatments])
    for patch, t in zip(bplot1['boxes'], treatments):
        patch.set_facecolor(treatment_color_map[t])
    # Set median lines thicker and black
    for median in bplot1['medians']:
        median.set(color='black', linewidth=2.5)
    axes1[idx, 0].set_title(f'Site {site} - skel_length_mm by Treatment')
    axes1[idx, 0].set_xlabel('site_treatment')
    axes1[idx, 0].set_ylabel('skel_length_mm')
    axes1[idx, 0].grid(axis='y')
    # Boxplot for nn_pred_body / conv_rate_mm_px
    data2 = [
        (site_df[site_df['site_treatment'] == t]['nn_pred_body'] / site_df[site_df['site_treatment'] == t]['conv_rate_mm_px']).dropna()
        for t in treatments
    ]
    bplot2 = axes1[idx, 1].boxplot(data2, patch_artist=True, labels=[treatment_display_names[t] for t in treatments])
    for patch, t in zip(bplot2['boxes'], treatments):
        patch.set_facecolor(treatment_color_map[t])
    # Set median lines thicker and black
    for median in bplot2['medians']:
        median.set(color='black', linewidth=2.5)
    axes1[idx, 1].set_title(f'Site {site} - nn_pred_body / conv_rate_mm_px by Treatment')
    axes1[idx, 1].set_xlabel('site_treatment')
    axes1[idx, 1].set_ylabel('nn_pred_body / conv_rate_mm_px')
    axes1[idx, 1].grid(axis='y')

# Add a legend for site_treatment colors
legend_handles = [Patch(facecolor=treatment_color_map[t], label=treatment_display_names[t]) for t in all_treatments]
fig1.legend(handles=legend_handles, title='site_treatment', loc='upper right')

# # Basic boxplot
# plt.tight_layout(rect=[0, 0, 0.95, 1])
# plt.savefig(os.path.join(output_dir, 'all_sites_boxplots.png'))
# plt.close()

# Variables to plot and their y-labels
panel_vars = [
    ('nn_pred_body', 'Body Length (mm)', lambda df: df['nn_pred_body'] / df['conv_rate_mm_px']),
    ('nn_pred_head', 'Head Width (mm)', lambda df: df['nn_pred_head'] / df['conv_rate_mm_px']),
    ('area', 'Area (mm²)', lambda df: df['area'] / (df['conv_rate_mm_px'] ** 2)),
]

# # --- New multi-panel plot for three variables ---
# fig2, axes2 = plt.subplots(n_sites, 3, figsize=(21, 5 * n_sites), squeeze=False)

# for idx, site in enumerate(site_numbers):
#     site_df = df[df['site_number'] == site]
#     treatments = [t for t in all_treatments if t in site_df['site_treatment'].unique()]
#     for col_idx, (var, ylabel, func) in enumerate(panel_vars):
#         data = [func(site_df[site_df['site_treatment'] == t]).dropna() for t in treatments]
#         bplot = axes2[idx, col_idx].boxplot(data, patch_artist=True, labels=[treatment_display_names[t] for t in treatments])
#         for patch, t in zip(bplot['boxes'], treatments):
#             patch.set_facecolor(treatment_color_map[t])
#         for median in bplot['medians']:
#             median.set(color='black', linewidth=2.5)
#         axes2[idx, col_idx].set_title(f'Site {site} - {ylabel} by Treatment')
#         axes2[idx, col_idx].set_xlabel('site_treatment')
#         axes2[idx, col_idx].set_ylabel(ylabel)
#         axes2[idx, col_idx].grid(axis='y')

# # Add a legend for site_treatment colors
# legend_handles2 = [Patch(facecolor=treatment_color_map[t], label=treatment_display_names[t]) for t in all_treatments]
# fig2.legend(handles=legend_handles2, title='site_treatment', loc='upper right')

# plt.tight_layout(rect=[0, 0, 0.95, 1])
# plt.savefig(os.path.join(output_dir, 'all_sites_boxplots_multi_panel.png'))
# plt.close(fig2)

# --- Assign pred_class color palette globally for all panels, using a different colormap ---
if 'pred_class' in df.columns:
    all_pred_classes = [pc for pc in df['pred_class'].unique() if any(df['pred_class'] == pc)]
    # Use a different colormap, e.g., 'Set2' for pred_class
    pred_class_palette = plt.cm.get_cmap('Set2', len(all_pred_classes))
    pred_class_color_map = {pc: pred_class_palette(i) for i, pc in enumerate(all_pred_classes)}
else:
    pred_class_color_map = {}

fig3, axes3 = plt.subplots(n_sites, 3, figsize=(21, 5 * n_sites), squeeze=False)

for idx, site in enumerate(site_numbers):
    site_df = df[df['site_number'] == site]
    treatments = [t for t in all_treatments if t in site_df['site_treatment'].unique()]
    pred_classes = site_df['pred_class'].unique() if 'pred_class' in site_df.columns else []
    # For each treatment, get pred_classes present in this site/treatment
    pairs = []
    labels = []
    colors = []
    for t in treatments:
        pcs = [pc for pc in pred_classes if pc != 'errors' and not site_df[(site_df['site_treatment'] == t) & (site_df['pred_class'] == pc)].empty]
        for pc in pcs:
            pairs.append((t, pc))
            labels.append(f"{treatment_display_names[t]}\n{pc}")
            colors.append(pred_class_color_map.get(pc, '#cccccc'))  # color by pred_class only
    # Now, for each variable, plot with no spacing between boxplots of the same treatment
    for col_idx, (var, ylabel, func) in enumerate(panel_vars):
        data = [func(site_df[(site_df['site_treatment'] == t) & (site_df['pred_class'] == pc)]).dropna() for t, pc in pairs]
        # Calculate positions: group by treatment, wider gap between treatments
        positions = []
        group_positions = []
        group_labels = []
        pos = 1
        last_t = None
        group_start = pos
        for i, (t, pc) in enumerate(pairs):
            if last_t is not None and t != last_t:
                # Place group tick at the center of the previous group
                group_center = (group_start + positions[-1]) / 2
                group_positions.append(group_center)
                group_labels.append(treatment_display_names[last_t])
                pos += 1.5  # wider gap between treatments
                group_start = pos
            positions.append(pos)
            pos += 1
            last_t = t
        # Add the last group tick
        if positions:
            group_center = (group_start + positions[-1]) / 2
            # Wrap long treatment names to two lines
            def wrap_label(label):
                if ' ' in label:
                    parts = label.split(' ')
                    return '\n'.join(parts)
                return label
            group_labels.append(wrap_label(treatment_display_names[last_t]))
            group_positions.append(group_center)
        bplot = axes3[idx, col_idx].boxplot(data, patch_artist=True, labels=labels, positions=positions, widths=0.8)
        axes3[idx, col_idx].set_xticks(group_positions)
        # Wrap all group labels
        axes3[idx, col_idx].set_xticklabels([wrap_label(lbl) for lbl in group_labels], rotation=0, ha='center', fontsize=11)
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        for median in bplot['medians']:
            median.set(color='black', linewidth=2.5)
        axes3[idx, col_idx].set_title(f'Site {site} - {ylabel} by Treatment & pred_class')
        axes3[idx, col_idx].set_xlabel('site_treatment')
        axes3[idx, col_idx].set_ylabel(ylabel)
        axes3[idx, col_idx].grid(axis='y')
# Add a legend for pred_class colors
if pred_class_color_map:
    legend_handles4 = [Patch(facecolor=pred_class_color_map[pc], label=str(pc)) for pc in all_pred_classes if pc != 'errors']
    fig3.legend(handles=legend_handles4, title='pred_class', loc='upper right')
else:
    legend_handles4 = []
plt.tight_layout(rect=[0, 0, 0.95, 1])
plt.savefig(os.path.join(output_dir, 'all_sites_boxplots_multi_panel_by_pred_class_grouped_predclasscolor.png'))
plt.close(fig3)

### --- New plot: All sites combined, only treatments ob, b1, b2 ---
if 'pred_class' in df.columns:
    selected_treatments = ['ob', 'b1', 'b2']
    selected_treatments = [t for t in selected_treatments if t in df['site_treatment'].unique()]
    selected_pred_classes = [pc for pc in df['pred_class'].unique() if pc != 'errors' and any(df['pred_class'] == pc)]
    pairs = []
    labels = []
    colors = []
    for t in selected_treatments:
        for pc in selected_pred_classes:
            subset = df[(df['site_treatment'] == t) & (df['pred_class'] == pc)]
            if not subset.empty:
                pairs.append((t, pc))
                labels.append(f"{treatment_display_names[t]}\n{pc}")
                colors.append(pred_class_color_map.get(pc, '#cccccc'))
    fig_selected, axes_selected = plt.subplots(1, 3, figsize=(21, 6), squeeze=False)
    for col_idx, (var, ylabel, func) in enumerate(panel_vars):
        data = [func(df[(df['site_treatment'] == t) & (df['pred_class'] == pc)]).dropna() for t, pc in pairs]
        positions = []
        group_positions = []
        group_labels = []
        pos = 1
        last_t = None
        group_start = pos
        for i, (t, pc) in enumerate(pairs):
            if last_t is not None and t != last_t:
                group_center = (group_start + positions[-1]) / 2
                group_positions.append(group_center)
                group_labels.append(treatment_display_names[last_t])
                pos += 1.5
                group_start = pos
            positions.append(pos)
            pos += 1
            last_t = t
        if positions:
            group_center = (group_start + positions[-1]) / 2
            def wrap_label(label):
                if ' ' in label:
                    parts = label.split(' ')
                    return '\n'.join(parts)
                return label
            group_labels.append(wrap_label(treatment_display_names[last_t]))
            group_positions.append(group_center)
        bplot = axes_selected[0, col_idx].boxplot(data, patch_artist=True, labels=labels, positions=positions, widths=0.8)
        axes_selected[0, col_idx].set_xticks(group_positions)
        axes_selected[0, col_idx].set_xticklabels([wrap_label(lbl) for lbl in group_labels], rotation=0, ha='center', fontsize=11)
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        for median in bplot['medians']:
            median.set(color='black', linewidth=2.5)
        axes_selected[0, col_idx].set_title(f'All Sites - {ylabel} by Treatment & pred_class (ob, b1, b2)')
        axes_selected[0, col_idx].set_xlabel('site_treatment')
        axes_selected[0, col_idx].set_ylabel(ylabel)
        axes_selected[0, col_idx].grid(axis='y')

# --- New plot: All sites combined, only treatments bd, ur, hf1, hf2 ---
if 'pred_class' in df.columns:
    selected_treatments2 = ['bd', 'ur', 'hf1', 'hf2']
    selected_treatments2 = [t for t in selected_treatments2 if t in df['site_treatment'].unique()]
    selected_pred_classes2 = [pc for pc in df['pred_class'].unique() if pc != 'errors' and any(df['pred_class'] == pc)]
    pairs2 = []
    labels2 = []
    colors2 = []
    for t in selected_treatments2:
        for pc in selected_pred_classes2:
            subset = df[(df['site_treatment'] == t) & (df['pred_class'] == pc)]
            if not subset.empty:
                pairs2.append((t, pc))
                labels2.append(f"{treatment_display_names[t]}\n{pc}")
                colors2.append(pred_class_color_map.get(pc, '#cccccc'))
    fig_selected2, axes_selected2 = plt.subplots(1, 3, figsize=(21, 6), squeeze=False)
    for col_idx, (var, ylabel, func) in enumerate(panel_vars):
        data2 = [func(df[(df['site_treatment'] == t) & (df['pred_class'] == pc)]).dropna() for t, pc in pairs2]
        positions2 = []
        group_positions2 = []
        group_labels2 = []
        pos2 = 1
        last_t2 = None
        group_start2 = pos2
        for i, (t, pc) in enumerate(pairs2):
            if last_t2 is not None and t != last_t2:
                group_center2 = (group_start2 + positions2[-1]) / 2
                group_positions2.append(group_center2)
                group_labels2.append(treatment_display_names[last_t2])
                pos2 += 1.5
                group_start2 = pos2
            positions2.append(pos2)
            pos2 += 1
            last_t2 = t
        if positions2:
            group_center2 = (group_start2 + positions2[-1]) / 2
            def wrap_label2(label):
                if ' ' in label:
                    parts = label.split(' ')
                    return '\n'.join(parts)
                return label
            group_labels2.append(wrap_label2(treatment_display_names[last_t2]))
            group_positions2.append(group_center2)
        bplot2 = axes_selected2[0, col_idx].boxplot(data2, patch_artist=True, labels=labels2, positions=positions2, widths=0.8)
        axes_selected2[0, col_idx].set_xticks(group_positions2)
        axes_selected2[0, col_idx].set_xticklabels([wrap_label2(lbl) for lbl in group_labels2], rotation=0, ha='center', fontsize=11)
        for patch, color in zip(bplot2['boxes'], colors2):
            patch.set_facecolor(color)
        for median in bplot2['medians']:
            median.set(color='black', linewidth=2.5)
        axes_selected2[0, col_idx].set_title(f'All Sites - {ylabel} by Treatment & pred_class (bd, ur, hf1, hf2)')
        axes_selected2[0, col_idx].set_xlabel('site_treatment')
        axes_selected2[0, col_idx].set_ylabel(ylabel)
        axes_selected2[0, col_idx].grid(axis='y')
    # Add legend for pred_class colors
    if pred_class_color_map:
        legend_handles_selected2 = [Patch(facecolor=pred_class_color_map[pc], label=str(pc)) for pc in selected_pred_classes2]
        fig_selected2.legend(handles=legend_handles_selected2, title='pred_class', loc='upper right')
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.savefig(os.path.join(output_dir, 'all_sites_combined_boxplots_multi_panel_by_pred_class_grouped_predclasscolor_bd_ur_hf1_hf2.png'))
    plt.close(fig_selected2)
if 'pred_class' in df.columns:
    # Get all treatments and pred_classes (excluding 'errors')
    combined_treatments = [t for t in all_treatments if t in df['site_treatment'].unique()]
    combined_pred_classes = [pc for pc in df['pred_class'].unique() if pc != 'errors' and any(df['pred_class'] == pc)]
    # Prepare pairs (treatment, pred_class) that exist in the data
    pairs = []
    labels = []
    colors = []
    for t in combined_treatments:
        for pc in combined_pred_classes:
            subset = df[(df['site_treatment'] == t) & (df['pred_class'] == pc)]
            if not subset.empty:
                pairs.append((t, pc))
                labels.append(f"{treatment_display_names[t]}\n{pc}")
                colors.append(pred_class_color_map.get(pc, '#cccccc'))
    fig_combined, axes_combined = plt.subplots(1, 3, figsize=(21, 6), squeeze=False)
    for col_idx, (var, ylabel, func) in enumerate(panel_vars):
        data = [func(df[(df['site_treatment'] == t) & (df['pred_class'] == pc)]).dropna() for t, pc in pairs]
        # Calculate positions: group by treatment, wider gap between treatments
        positions = []
        group_positions = []
        group_labels = []
        pos = 1
        last_t = None
        group_start = pos
        for i, (t, pc) in enumerate(pairs):
            if last_t is not None and t != last_t:
                group_center = (group_start + positions[-1]) / 2
                group_positions.append(group_center)
                group_labels.append(treatment_display_names[last_t])
                pos += 1.5
                group_start = pos
            positions.append(pos)
            pos += 1
            last_t = t
        if positions:
            group_center = (group_start + positions[-1]) / 2
            def wrap_label(label):
                if ' ' in label:
                    parts = label.split(' ')
                    return '\n'.join(parts)
                return label
            group_labels.append(wrap_label(treatment_display_names[last_t]))
            group_positions.append(group_center)
        bplot = axes_combined[0, col_idx].boxplot(data, patch_artist=True, labels=labels, positions=positions, widths=0.8)
        axes_combined[0, col_idx].set_xticks(group_positions)
        axes_combined[0, col_idx].set_xticklabels([wrap_label(lbl) for lbl in group_labels], rotation=0, ha='center', fontsize=11)
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        for median in bplot['medians']:
            median.set(color='black', linewidth=2.5)
        axes_combined[0, col_idx].set_title(f'All Sites - {ylabel} by Treatment & pred_class')
        axes_combined[0, col_idx].set_xlabel('site_treatment')
        axes_combined[0, col_idx].set_ylabel(ylabel)
        axes_combined[0, col_idx].grid(axis='y')
    # Add legend for pred_class colors
    if pred_class_color_map:
        legend_handles_combined = [Patch(facecolor=pred_class_color_map[pc], label=str(pc)) for pc in combined_pred_classes if pc != 'errors']
        fig_combined.legend(handles=legend_handles_combined, title='pred_class', loc='upper right')
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.savefig(os.path.join(output_dir, 'all_sites_combined_boxplots_multi_panel_by_pred_class_grouped_predclasscolor.png'))
    plt.close(fig_combined)



# --- New plot: Stacked barplot of pred_class counts per treatment for each site ---
if 'pred_class' in df.columns:
    fig_bar, axes_bar = plt.subplots(1, n_sites, figsize=(7 * n_sites, 6), sharey=True)
    if n_sites == 1:
        axes_bar = [axes_bar]
    for idx, site in enumerate(site_numbers):
        site_df = df[df['site_number'] == site]
        treatments = [t for t in all_treatments if t in site_df['site_treatment'].unique()]
        # Count pred_class for each treatment
        count_df = site_df[site_df['pred_class'] != 'errors'].groupby(['site_treatment', 'pred_class']).size().unstack(fill_value=0)
        # Ensure all treatments and pred_classes are present in the index/columns
        count_df = count_df.reindex(index=treatments, columns=all_pred_classes, fill_value=0)
        # Plot stacked bar
        bottom = None
        for pc in all_pred_classes:
            axes_bar[idx].bar(
                [treatment_display_names[t] for t in treatments],
                count_df.loc[treatments, pc],
                label=str(pc),
                color=pred_class_color_map.get(pc, '#cccccc'),
                bottom=bottom
            )
            if bottom is None:
                bottom = count_df.loc[treatments, pc].copy()
            else:
                bottom += count_df.loc[treatments, pc]
        axes_bar[idx].set_title(f'Site {site} - pred_class counts by Treatment')
        axes_bar[idx].set_xlabel('Treatment')
        axes_bar[idx].set_ylabel('Count')
        axes_bar[idx].tick_params(axis='x', rotation=30)
    fig_bar.legend(handles=[Patch(facecolor=pred_class_color_map[pc], label=str(pc)) for pc in all_pred_classes if pc != 'errors'],
                  title='pred_class', loc='upper right')
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.savefig(os.path.join(output_dir, 'stacked_barplot_predclass_by_treatment.png'))
    plt.close(fig_bar)