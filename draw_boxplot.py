import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
from scipy.stats import ttest_ind
from matplotlib.ticker import MaxNLocator

# Font size configuration
TITLE_FONTSIZE = 18
AXIS_LABEL_FONTSIZE = 16
TICK_LABEL_FONTSIZE = 14
AXIS_LINEWIDTH = 1.5
BOX_LINEWIDTH = 1.75
# SIGNIFICANCE_FONTSIZE = 12

FIGURE_WIDTH = 8
FIGURE_HEIGHT = 5

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']


def plot_box_by_window_length(input_dir, start_time_min, step_interval_min, max_window,
                              use_ml_prediction=False, model_dir=None, model_type="rf",
                              verbose=False):
    """
    Fixed starting point, compare values under different window lengths
    - Each window length gets one x-coordinate (1~max_window)
    - Draw sighted and blind boxplots under each x-coordinate, with raw data points overlaid (top layer)
    - Perform t-tests and display significance levels on the plot

    Parameters:
    - input_dir: Directory containing sliding window data
    - start_time_min: Fixed starting time point
    - step_interval_min: Step interval in minutes
    - max_window: Maximum window length to include
    - use_ml_prediction: If True, use ML model predictions; if False, use Dark%
    - model_dir: Directory containing trained models (required if use_ml_prediction=True)
    - model_type: Type of ML model ("rf" for Random Forest)
    - verbose: Print debug information
    """
    if use_ml_prediction:
        if model_dir is None:
            raise ValueError("model_dir must be provided when use_ml_prediction=True")
        df = load_ml_predictions_by_window(model_dir, model_type,
                                           start_time_min, step_interval_min, max_window)
        y_label = "ML model–predicted probability"
        title_suffix = "Multi-behavior feature assessment"
        figure_name = 'Fig. 1(c)_ori.tif'
        title_y_offset = 1.15
        legend_y_offset = 1.2
    else:
        df = load_dark_percent_by_window(input_dir, start_time_min, step_interval_min, max_window)
        y_label = "Time in dark (%)"
        title_suffix = "Single-behavior feature assessment"
        figure_name = 'Fig. 1(a)_ori.tif'
        title_y_offset = 1.05
        legend_y_offset = 1.1

    if verbose:
        print(df.head())

    def get_significance_stars(p_value):
        """Convert p-value to significance stars"""
        if p_value < 0.001:
            return '***'
        elif p_value < 0.01:
            return '**'
        elif p_value < 0.05:
            return '*'
        else:
            return 'NS'

    plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    ax = sns.boxplot(
        x='Window', y='Value', hue='Group', data=df, palette='Set2', showcaps=True,
        boxprops={'zorder': 1, 'linewidth': BOX_LINEWIDTH, 'edgecolor': 'black'},
        whiskerprops={'linewidth': BOX_LINEWIDTH, 'color': 'black'},
        capprops={'linewidth': BOX_LINEWIDTH, 'color': 'black'},
        medianprops={'linewidth': BOX_LINEWIDTH, 'color': 'black'},
        showfliers=False
    )
    # sns.stripplot(x='Window', y='Value', hue='Group', data=df, dodge=True, jitter=False, alpha=0.7, zorder=2,
    #               palette='Set2', marker='o', edgecolor='black', linewidth=0.5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(AXIS_LINEWIDTH)
    ax.spines['bottom'].set_linewidth(AXIS_LINEWIDTH)

    # Perform t-tests and add significance annotations
    windows = sorted(df['Window'].unique())
    print(f"\nT-test results for window length comparison ({title_suffix}, Start at {start_time_min} min):")
    print("Window\tSighted_mean±std\tBlind_mean±std\tt-stat\tp-value\tSignificance")
    print("-" * 80)

    for i, window in enumerate(windows):
        sighted_data = df[(df['Window'] == window) & (df['Group'] == 'Sighted')]['Value']
        blind_data = df[(df['Window'] == window) & (df['Group'] == 'Blind')]['Value']

        if len(sighted_data) > 0 and len(blind_data) > 0:
            t_stat, p_value = ttest_ind(sighted_data, blind_data)
            significance = get_significance_stars(p_value)

            # Print results
            sighted_mean, sighted_std = sighted_data.mean(), sighted_data.std()
            blind_mean, blind_std = blind_data.mean(), blind_data.std()
            print(
                f"{window}\t{sighted_mean:.2f}±{sighted_std:.2f}\t\t{blind_mean:.2f}±{blind_std:.2f}\t\t{t_stat:.3f}\t{p_value:.4f}\t{significance}")

            # Add significance annotation to plot
            y_max = max(sighted_data.max(), blind_data.max())
            y_pos = y_max + (df['Value'].max() - df['Value'].min()) * 0.05
            # ax.text(i, y_pos, significance, ha='center', va='bottom', fontsize=SIGNIFICANCE_FONTSIZE, fontweight='bold')

    plt.title(title_suffix, fontsize=TITLE_FONTSIZE, y=title_y_offset)
    plt.xlabel("Window length (min; start time = 0 min)", fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel(y_label, fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
    plt.legend(bbox_to_anchor=(1.0, legend_y_offset), loc='upper right', frameon=False)
    # plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(figure_name, dpi=300, bbox_inches='tight')
    plt.show()


def plot_box_by_start_time(input_dir, window_length, step_interval_min,
                           use_ml_prediction=False, model_dir=None, model_type="rf",
                           max_window_length=None, verbose=False):
    """
    Fixed window length, compare values under different starting points
    - X-axis represents start time
    - Draw sighted and blind boxplots under each x-coordinate, with raw data points overlaid (top layer)
    - Perform t-tests and display significance levels on the plot

    Parameters:
    - input_dir: Directory containing sliding window data
    - window_length: Fixed window length in minutes
    - step_interval_min: Step interval in minutes
    - use_ml_prediction: If True, use ML model predictions; if False, use Dark%
    - model_dir: Directory containing trained models (required if use_ml_prediction=True)
    - model_type: Type of ML model ("rf" for Random Forest)
    - max_window_length: Maximum total length (start_time + window_length). If None, no limit is applied.
    - verbose: Print debug information
    """
    if use_ml_prediction:
        if model_dir is None:
            raise ValueError("model_dir must be provided when use_ml_prediction=True")
        df = load_ml_predictions_by_start_time(model_dir, model_type,
                                               window_length, step_interval_min)
        y_label = "ML model–predicted probability"
        title_suffix = "Multi-behavior feature assessment"
        figure_name = 'Fig. 1(d)_ori.tif'
        title_y_offset = 1.15
        legend_y_offset = 1.2
    else:
        df = load_dark_percent_by_start_time(input_dir, window_length, step_interval_min)
        y_label = "Time in dark (%)"
        title_suffix = "Single-behavior feature assessment"
        figure_name = 'Fig. 1(b)_ori.tif'
        title_y_offset = 1.05
        legend_y_offset = 1.1

    # Filter data based on max_window_length if provided
    if max_window_length is not None:
        # Convert 'Start' column to float for comparison
        df['Start_float'] = df['Start'].astype(float)
        # Filter out data points where start_time + window_length > max_window_length
        df = df[df['Start_float'] + window_length <= max_window_length]
        # Drop the temporary column
        df = df.drop('Start_float', axis=1)

        if verbose:
            print(
                f"Filtered data to include only start times where start_time + {window_length} <= {max_window_length}")
            print(f"Remaining data points: {len(df)}")

    if verbose:
        print(df.head())

    # Check if there's any data left after filtering
    if len(df) == 0:
        print("Warning: No data points remain after filtering with max_window_length constraint.")
        return

    def get_significance_stars(p_value):
        """Convert p-value to significance stars"""
        if p_value < 0.001:
            return '***'
        elif p_value < 0.01:
            return '**'
        elif p_value < 0.05:
            return '*'
        else:
            return 'NS'

    plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    ax = sns.boxplot(
        x='Start', y='Value', hue='Group', data=df, palette='Set2', showcaps=True,
        boxprops={'zorder': 1, 'linewidth': BOX_LINEWIDTH, 'edgecolor': 'black'},
        whiskerprops={'linewidth': BOX_LINEWIDTH, 'color': 'black'},
        capprops={'linewidth': BOX_LINEWIDTH, 'color': 'black'},
        medianprops={'linewidth': BOX_LINEWIDTH, 'color': 'black'},
        showfliers=False
    )
    # sns.stripplot(x='Start', y='Value', hue='Group', data=df, dodge=True, jitter=False, alpha=0.7, zorder=2,
    #               palette='Set2', marker='o', edgecolor='black', linewidth=0.5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(AXIS_LINEWIDTH)
    ax.spines['bottom'].set_linewidth(AXIS_LINEWIDTH)

    # Perform t-tests and add significance annotations
    start_times = sorted(df['Start'].unique(), key=lambda x: float(x))
    max_length_info = f" (max_length={max_window_length}min)" if max_window_length is not None else ""
    print(f"\nT-test results for start time comparison ({title_suffix}, Window = {window_length}min{max_length_info}):")
    print("Start_time\tSighted_mean±std\tBlind_mean±std\tt-stat\tp-value\tSignificance")
    print("-" * 85)

    for i, start_time in enumerate(start_times):
        sighted_data = df[(df['Start'] == start_time) & (df['Group'] == 'Sighted')]['Value']
        blind_data = df[(df['Start'] == start_time) & (df['Group'] == 'Blind')]['Value']

        if len(sighted_data) > 0 and len(blind_data) > 0:
            t_stat, p_value = ttest_ind(sighted_data, blind_data)
            significance = get_significance_stars(p_value)

            # Print results
            sighted_mean, sighted_std = sighted_data.mean(), sighted_data.std()
            blind_mean, blind_std = blind_data.mean(), blind_data.std()
            print(
                f"{start_time}\t\t{sighted_mean:.2f}±{sighted_std:.2f}\t\t{blind_mean:.2f}±{blind_std:.2f}\t\t{t_stat:.3f}\t{p_value:.4f}\t{significance}")

            # Add significance annotation to plot
            y_max = max(sighted_data.max(), blind_data.max())
            y_pos = y_max + (df['Value'].max() - df['Value'].min()) * 0.05
            # ax.text(i, y_pos, significance, ha='center', va='bottom', fontsize=SIGNIFICANCE_FONTSIZE, fontweight='bold')

    # Update x-axis label to include max_window_length info if applicable
    plt.title(title_suffix, fontsize=TITLE_FONTSIZE, y=title_y_offset)
    plt.xlabel(f"Start time (min; window length = {window_length} min)", fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel(y_label, fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
    ax.set_xticks(range(0, max_window_length - window_length + 1, 1))
    ax.set_xticklabels([str(i) for i in range(0, max_window_length - window_length + 1, 1)])
    if use_ml_prediction is False:
        ax.set_yticks(range(0, 101, 20))
    plt.legend(bbox_to_anchor=(1.0, legend_y_offset), loc='upper right', frameon=False)
    # plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(figure_name, dpi=300, bbox_inches='tight')
    plt.show()


def load_dark_percent_by_window(input_dir, start_time_min, step_interval_min, max_window):
    """Load Dark% data for window length comparison"""
    all_records = []
    start_index = int(start_time_min / step_interval_min)

    for fname in os.listdir(input_dir):
        if not fname.endswith(".txt"):
            continue

        parts = fname.split('_')
        if len(parts) < 5:
            continue
        group = parts[2].lower()
        if group not in ['normal', 'blind']:
            continue

        match = re.search(r"W(\d+)min", fname)
        if not match:
            continue
        window_length = int(match.group(1))
        if window_length > max_window:
            continue

        step_str = f"S{step_interval_min:.2f}min"
        if step_str not in fname:
            continue

        path = os.path.join(input_dir, fname)
        data = np.loadtxt(path)
        if start_index >= len(data):
            continue

        dark_percent = data[start_index, 0]

        # Map 'normal' to 'Sighted' for display purposes
        display_group = 'Sighted' if group == 'normal' else group.capitalize()

        all_records.append({
            'Group': display_group,
            'Window': window_length,
            'Value': dark_percent
        })

    return pd.DataFrame(all_records)


def load_dark_percent_by_start_time(input_dir, window_length, step_interval_min):
    """Load Dark% data for start time comparison"""
    all_records = []
    window_str = f"W{window_length}min"
    step_str = f"S{step_interval_min:.2f}min"

    for fname in os.listdir(input_dir):
        if not fname.endswith(".txt"):
            continue
        if window_str not in fname or step_str not in fname:
            continue

        parts = fname.split('_')
        if len(parts) < 5:
            continue
        group = parts[2].lower()
        if group not in ['normal', 'blind']:
            continue

        path = os.path.join(input_dir, fname)
        data = np.loadtxt(path)

        for i in range(len(data)):
            start_time = i * step_interval_min
            dark_percent = data[i, 0]

            # Map 'normal' to 'Sighted' for display purposes
            display_group = 'Sighted' if group == 'normal' else group.capitalize()

            all_records.append({
                'Group': display_group,
                'Start': f"{start_time:.1f}",
                'Value': dark_percent
            })

    return pd.DataFrame(all_records)


def load_ml_predictions_by_window(model_dir, model_type,
                                  start_time_min, step_interval_min, max_window):
    """
    Load ML prediction data for window length comparison

    Parameters:
    - model_dir: Directory containing trained models
    - model_type: Type of ML model ("rf" for Random Forest)
    - start_time_min: Fixed starting time point
    - step_interval_min: Step interval in minutes
    - max_window: Maximum window length to include
    """
    all_records = []
    start_index = int(start_time_min / step_interval_min)

    # Load predictions for each window length from 1 to max_window
    for window_length in range(1, max_window + 1):
        predictions_path = os.path.join(model_dir, model_type, f"predictions_{model_type}_w{window_length}min_s{step_interval_min}min.csv")
        if not os.path.exists(predictions_path):
            print(f"Warning: Predictions file not found: {predictions_path}")
            continue

        df_pred = pd.read_csv(predictions_path)

        # Filter for the specific start time (sample_index)
        df_filtered = df_pred[df_pred['sample_index'] == start_index]

        for _, row in df_filtered.iterrows():
            # Map 'normal' to 'Sighted' for display purposes
            display_group = 'Sighted' if row['group'].lower() == 'normal' else row['group'].capitalize()

            all_records.append({
                'Group': display_group,
                'Window': window_length,  # This represents the window length of the model
                'Value': row['predicted_prob']
            })

    return pd.DataFrame(all_records)


def load_ml_predictions_by_start_time(model_dir, model_type,
                                      window_length, step_interval_min):
    """
    Load ML prediction data for start time comparison

    Parameters:
    - model_dir: Directory containing trained models
    - model_type: Type of ML model ("rf" for Random Forest)
    - window_length: Window length in minutes (used to find the correct model file)
    - step_interval_min: Step interval in minutes
    """
    predictions_path = os.path.join(model_dir, model_type, f"predictions_{model_type}_w{window_length}min_s{step_interval_min}min.csv")

    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    df_pred = pd.read_csv(predictions_path)
    all_records = []

    for _, row in df_pred.iterrows():
        start_time = row['sample_index'] * step_interval_min

        # Map 'normal' to 'Sighted' for display purposes
        display_group = 'Sighted' if row['group'].lower() == 'normal' else row['group'].capitalize()

        all_records.append({
            'Group': display_group,
            'Start': f"{start_time:.1f}",
            'Value': row['predicted_prob']
        })

    return pd.DataFrame(all_records)


def plot_age_distribution_boxplot(excel_path="Mice information/Mice information.xlsx"):
    # Read Excel file
    df = pd.read_excel(excel_path)

    # Extract age (weeks)
    df["Age(weeks)"] = df["Age"].astype(str).str.extract(r"(\d+)").astype(float)

    # Set up canvas
    plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

    # Boxplot
    sns.boxplot(x="Blind/Normal", y="Age(weeks)", data=df,
                palette={"Blind": "coral", "Sighted": "mediumseagreen"},
                width=0.6, boxprops=dict(alpha=0.7))

    # Scatter plot (jittered to prevent overlap)
    sns.stripplot(x="Blind/Normal", y="Age(weeks)", data=df,
                  color="black", size=5, jitter=True, alpha=0.5)

    plt.title("Age Distribution", fontsize=TITLE_FONTSIZE)
    plt.xlabel("Type", fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel("Age (weeks)", fontsize=AXIS_LABEL_FONTSIZE)
    # plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()