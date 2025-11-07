import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

# Font size configuration (matching draw_boxplot.py style)
TITLE_FONTSIZE = 18
AXIS_LABEL_FONTSIZE = 16
TICK_LABEL_FONTSIZE = 14
LEGEND_FONTSIZE = 14
AXIS_LINEWIDTH = 1.5
BAR_WIDTH = 0.35

FIGURE_WIDTH = 8
FIGURE_HEIGHT = 5

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']


def plot_auc_by_window_length(input_dir, ml_model_dir, max_window_length=10,
                              step_interval_min=1.0, model_type="rf",
                              single_feature_index=0, verbose=False,
                              roc_window_length=None):
    """
    Plot AUC values for single-feature and multi-feature models across different window lengths
    Optionally plot ROC curves for a specific window length

    Parameters:
    - input_dir: Directory containing sliding window feature data
    - ml_model_dir: Directory containing trained ML models
    - max_window_length: Maximum window length to analyze
    - step_interval_min: Step interval in minutes
    - model_type: Type of ML model ("rf" for Random Forest)
    - single_feature_index: Index of the feature to use for single-feature analysis (0 = Dark%)
    - verbose: Print debug information
    - roc_window_length: If specified, plot ROC curves for this window length (in addition to AUC bar plot)
    """

    window_lengths = list(range(1, max_window_length + 1))
    single_feature_aucs = []
    multi_feature_aucs = []
    single_feature_stds = []
    multi_feature_stds = []
    single_feature_accuracies = []
    multi_feature_accuracies = []

    print(f"Calculating AUC values for window lengths 1-{max_window_length} minutes...")

    for window_length in window_lengths:
        print(f"\nProcessing window length = {window_length} min")

        # Calculate single-feature AUC (using specified feature, default: Dark%)
        single_auc = calculate_single_feature_auc(
            input_dir, window_length, step_interval_min, single_feature_index, verbose
        )
        single_feature_aucs.append(single_auc)
        single_feature_stds.append(0.0)  # No std for single calculation

        # Calculate single-feature accuracy
        single_acc = calculate_single_feature_accuracy(
            input_dir, window_length, step_interval_min, single_feature_index, verbose
        )
        single_feature_accuracies.append(single_acc)

        # Get multi-feature AUC from trained models
        multi_auc = get_multi_feature_auc_from_models(
            ml_model_dir, model_type, window_length, step_interval_min, verbose
        )
        multi_feature_aucs.append(multi_auc)
        multi_feature_stds.append(0.0)  # No std for single calculation

        # Get multi-feature accuracy from trained models
        multi_acc = get_multi_feature_accuracy_from_models(
            ml_model_dir, model_type, window_length, step_interval_min, verbose
        )
        multi_feature_accuracies.append(multi_acc)

        if verbose:
            print(f"  Single-feature AUC: {single_auc:.3f}")
            print(f"  Multi-feature AUC: {multi_auc:.3f}")

    # Create the AUC bar plot
    create_auc_bar_plot(window_lengths, single_feature_aucs, multi_feature_aucs,
                        single_feature_stds, multi_feature_stds, single_feature_index)

    # Print accuracy summary
    print("\n=== Accuracy Summary ===")
    print("Window\tSingle-feature Accuracy\tMulti-feature Accuracy")
    print("-" * 55)
    for i, w in enumerate(window_lengths):
        print(f"{w} min\t{single_feature_accuracies[i]:.3f}\t\t{multi_feature_accuracies[i]:.3f}")

    # If ROC window length is specified, plot ROC curves for that window length
    if roc_window_length is not None:
        if roc_window_length in window_lengths:
            print(f"\nPlotting ROC curves for window length = {roc_window_length} min")
            plot_roc_curves(input_dir, ml_model_dir, roc_window_length,
                            step_interval_min, model_type, single_feature_index, verbose)
        else:
            print(f"Warning: ROC window length {roc_window_length} is not in the range 1-{max_window_length}")


def plot_roc_curves(input_dir, ml_model_dir, window_length, step_interval_min,
                    model_type, single_feature_index, verbose=False):
    """
    Plot ROC curves for both single-feature and multi-feature models for a specific window length

    Parameters:
    - input_dir: Directory containing sliding window feature data
    - ml_model_dir: Directory containing trained ML models
    - window_length: Window length in minutes
    - step_interval_min: Step interval in minutes
    - model_type: Type of ML model ("rf" for Random Forest)
    - single_feature_index: Index of the feature to use for single-feature analysis
    - verbose: Print debug information
    """

    # Feature names for legend
    feature_names = [
        "Time in dark (%)", "Light distance (cm)", "Dark distance (cm)",
        "Light→Dark crossings", "Immobile time light (s)", "Immobile time dark (s)",
        "Light avg speed (cm/s)", "Latency to dark SD", "Latency to light SD"
    ]

    single_feature_name = feature_names[single_feature_index]

    # Get single-feature ROC data
    single_fpr, single_tpr, single_auc, single_thresholds = get_single_feature_roc_data(
        input_dir, window_length, step_interval_min, single_feature_index, verbose
    )

    # Get multi-feature ROC data
    multi_fpr, multi_tpr, multi_auc, multi_thresholds = get_multi_feature_roc_data(
        ml_model_dir, model_type, window_length, step_interval_min, verbose
    )

    if single_fpr is None or multi_fpr is None:
        print("Error: Could not generate ROC curves - missing data")
        return

    # Calculate accuracies for this specific window length
    single_acc = calculate_single_feature_accuracy(
        input_dir, window_length, step_interval_min, single_feature_index, verbose
    )
    multi_acc = get_multi_feature_accuracy_from_models(
        ml_model_dir, model_type, window_length, step_interval_min, verbose
    )

    # Create ROC curve plot
    plt.figure(figsize=(FIGURE_WIDTH - 1, FIGURE_HEIGHT))

    # Plot diagonal line (random chance)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, label='Random (AUC = 0.50)')

    # Plot single-feature ROC curve
    plt.plot(single_fpr, single_tpr, color='#2c3e50', linewidth=2.5, alpha=0.8,
             label=f'Single-feature (AUC = {single_auc:.3f})')

    # Plot multi-feature ROC curve
    plt.plot(multi_fpr, multi_tpr, color='#e74c3c', linewidth=2.5, alpha=0.8,
             label=f'Multi-feature (AUC = {multi_auc:.3f})')

    # Find and plot optimal points using Youden Index
    single_optimal_data = find_optimal_point(single_fpr, single_tpr, single_thresholds)
    multi_optimal_data = find_optimal_point(multi_fpr, multi_tpr, multi_thresholds)

    if single_optimal_data is not None:
        opt_fpr, opt_tpr, opt_threshold, opt_sensitivity, opt_specificity = single_optimal_data
        plt.plot(opt_fpr, opt_tpr, 'ko', markersize=8, markerfacecolor='black',
                 markeredgecolor='black', markeredgewidth=0)
        plt.text(opt_fpr + 0.02, opt_tpr - 0.18,
                 f'Threshold: {opt_threshold:.3f}\nSensitivity: {opt_sensitivity:.3f}\nSpecificity: {opt_specificity:.3f}',
                 fontsize=11, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    if multi_optimal_data is not None:
        opt_fpr, opt_tpr, opt_threshold, opt_sensitivity, opt_specificity = multi_optimal_data
        plt.plot(opt_fpr, opt_tpr, 'ro', markersize=8, markerfacecolor='red',
                 markeredgecolor='red', markeredgewidth=0)
        plt.text(opt_fpr - 0.10, opt_tpr + 0.08,
                 f'Threshold: {opt_threshold:.3f}\nSensitivity: {opt_sensitivity:.3f}\nSpecificity: {opt_specificity:.3f}',
                 fontsize=11, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Customize the plot
    plt.xlabel('False positive rate', fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel('True positive rate', fontsize=AXIS_LABEL_FONTSIZE)
    plt.title(f'ROC curves (window length = {window_length} min)', fontsize=TITLE_FONTSIZE, y=1.05)

    # Set axis properties
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks(fontsize=TICK_LABEL_FONTSIZE)
    plt.yticks(fontsize=TICK_LABEL_FONTSIZE)

    # Customize appearance
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_linewidth(AXIS_LINEWIDTH)
    plt.gca().spines['bottom'].set_linewidth(AXIS_LINEWIDTH)

    # Add legend
    plt.legend(bbox_to_anchor=(1.0, 0.1), loc='lower right', frameon=False, fontsize=LEGEND_FONTSIZE)

    # Add grid
    # plt.grid(True, alpha=0.3)

    # Tight layout and save
    plt.tight_layout()
    plt.savefig('Fig.2 (b)_ori.tif', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nROC curve analysis for {window_length}-minute window:")
    print(f"Single-feature AUC ({single_feature_name}): {single_auc:.3f}")
    print(f"Multi-feature AUC: {multi_auc:.3f}")
    print(f"AUC improvement: {multi_auc - single_auc:+.3f}")
    print(f"Single-feature Accuracy: {single_acc:.3f}")
    print(f"Multi-feature Accuracy: {multi_acc:.3f}")

    # Print optimal point information
    if single_optimal_data is not None:
        _, _, opt_threshold, opt_sensitivity, opt_specificity = single_optimal_data
        print(f"\nSingle-feature optimal point:")
        print(f"  Threshold: {opt_threshold:.3f}")
        print(f"  Sensitivity: {opt_sensitivity:.3f}")
        print(f"  Specificity: {opt_specificity:.3f}")

    if multi_optimal_data is not None:
        _, _, opt_threshold, opt_sensitivity, opt_specificity = multi_optimal_data
        print(f"\nMulti-feature optimal point:")
        print(f"  Threshold: {opt_threshold:.3f}")
        print(f"  Sensitivity: {opt_sensitivity:.3f}")
        print(f"  Specificity: {opt_specificity:.3f}")


def get_single_feature_roc_data(input_dir, window_length, step_interval_min,
                                feature_index, verbose=False):
    """
    Get ROC curve data for single feature

    Parameters:
    - input_dir: Directory containing sliding window feature data
    - window_length: Window length in minutes
    - step_interval_min: Step interval in minutes
    - feature_index: Index of the feature to use
    - verbose: Print debug information

    Returns:
    - fpr: False positive rates
    - tpr: True positive rates
    - auc: AUC value
    - thresholds: Threshold values for ROC curve points
    """

    # Load all data for this window length
    X, y, groups = load_feature_data(input_dir, window_length, step_interval_min)

    if len(X) == 0:
        print(f"Warning: No data found for window length {window_length} min")
        return None, None, None, None

    # Extract single feature values
    single_feature_values = X[:, feature_index]

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y, single_feature_values)
    auc = roc_auc_score(y, single_feature_values)

    # If AUC < 0.5, flip the curve (feature might be inversely related)
    if auc < 0.5:
        fpr, tpr, thresholds = roc_curve(y, -single_feature_values)
        auc = 1.0 - auc
        # Adjust thresholds for the flipped curve
        thresholds = -thresholds

    if verbose:
        print(f"  Single-feature ROC: {len(fpr)} points, AUC = {auc:.3f}")

    return fpr, tpr, auc, thresholds


def get_multi_feature_roc_data(ml_model_dir, model_type, window_length,
                               step_interval_min, verbose=False):
    """
    Get ROC curve data from trained multi-feature models

    Parameters:
    - ml_model_dir: Directory containing trained models
    - model_type: Type of model ("rf" for Random Forest)
    - window_length: Window length in minutes
    - step_interval_min: Step interval in minutes
    - verbose: Print debug information

    Returns:
    - fpr: False positive rates
    - tpr: True positive rates
    - auc: AUC value
    - thresholds: Threshold values for ROC curve points
    """

    # Load predictions from all folds
    predictions_path = os.path.join(ml_model_dir, model_type,
                                    f"predictions_{model_type}_w{window_length}min_s{step_interval_min}min.csv")

    if not os.path.exists(predictions_path):
        print(f"Warning: Predictions file not found: {predictions_path}")
        return None, None, None, None

    df_pred = pd.read_csv(predictions_path)

    if len(df_pred) == 0:
        print(f"Warning: No predictions found for window length {window_length}")
        return None, None, None, None

    # Calculate ROC curve using all predictions
    fpr, tpr, thresholds = roc_curve(df_pred['true_label'], df_pred['predicted_prob'])
    auc = roc_auc_score(df_pred['true_label'], df_pred['predicted_prob'])

    if verbose:
        print(f"  Multi-feature ROC: {len(fpr)} points, AUC = {auc:.3f}")
        print(f"  Found {len(df_pred)} total predictions")

    return fpr, tpr, auc, thresholds


def find_optimal_point(fpr, tpr, thresholds):
    """
    Find optimal point on ROC curve using Youden Index (maximizing sensitivity + specificity - 1)

    Parameters:
    - fpr: False positive rates
    - tpr: True positive rates
    - thresholds: Threshold values

    Returns:
    - tuple: (optimal_fpr, optimal_tpr, optimal_threshold, sensitivity, specificity)
    """
    if fpr is None or tpr is None or thresholds is None:
        return None

    # Calculate Youden Index for each point
    youden_index = tpr - fpr  # TPR - FPR = Sensitivity + Specificity - 1
    # g_mean = np.sqrt(tpr * (1 - fpr))

    # Find the index with maximum Youden Index
    optimal_idx = np.argmax(youden_index)
    # optimal_idx = np.argmax(g_mean)

    optimal_fpr = fpr[optimal_idx]
    optimal_tpr = tpr[optimal_idx]
    optimal_threshold = thresholds[optimal_idx]
    optimal_sensitivity = optimal_tpr  # TPR = Sensitivity
    optimal_specificity = 1 - optimal_fpr  # 1 - FPR = Specificity

    return optimal_fpr, optimal_tpr, optimal_threshold, optimal_sensitivity, optimal_specificity


def calculate_single_feature_auc(input_dir, window_length, step_interval_min,
                                 feature_index=0, verbose=False):
    """
    Calculate AUC for single feature using all data

    Parameters:
    - input_dir: Directory containing sliding window feature data
    - window_length: Window length in minutes
    - step_interval_min: Step interval in minutes
    - feature_index: Index of the feature to use (0=Dark%, 1=Light distance, etc.)
    - verbose: Print debug information

    Returns:
    - auc: AUC value for the single feature
    """

    # Load all data for this window length
    X, y, groups = load_feature_data(input_dir, window_length, step_interval_min)

    if len(X) == 0:
        print(f"Warning: No data found for window length {window_length} min")
        return 0.0

    # Extract single feature values
    single_feature_values = X[:, feature_index]

    # Calculate AUC directly using the feature values as scores
    # Higher feature values should correspond to one class
    auc = roc_auc_score(y, single_feature_values)

    # If AUC < 0.5, flip it (the feature might be inversely related to the class)
    if auc < 0.5:
        auc = 1.0 - auc

    return auc


def calculate_single_feature_accuracy(input_dir, window_length, step_interval_min,
                                      feature_index=0, verbose=False):
    """
    Calculate accuracy for single feature using optimal threshold

    Parameters:
    - input_dir: Directory containing sliding window feature data
    - window_length: Window length in minutes
    - step_interval_min: Step interval in minutes
    - feature_index: Index of the feature to use (0=Dark%, 1=Light distance, etc.)
    - verbose: Print debug information

    Returns:
    - accuracy: Accuracy value for the single feature
    """

    # Load all data for this window length
    X, y, groups = load_feature_data(input_dir, window_length, step_interval_min)

    if len(X) == 0:
        print(f"Warning: No data found for window length {window_length} min")
        return 0.0

    # Extract single feature values
    single_feature_values = X[:, feature_index]

    # Check if we need to flip the feature (if AUC < 0.5)
    auc = roc_auc_score(y, single_feature_values)
    flipped = False
    if auc < 0.5:
        # Flip the feature values
        single_feature_values = -single_feature_values
        flipped = True
        auc = 1.0 - auc

    # Calculate ROC curve to find optimal threshold
    fpr, tpr, thresholds = roc_curve(y, single_feature_values)

    # Find optimal threshold using Youden Index
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]

    # Calculate predictions using optimal threshold
    predictions = (single_feature_values >= optimal_threshold).astype(int)

    # Calculate accuracy
    accuracy = accuracy_score(y, predictions)

    if verbose:
        print(
            f"  Single feature {feature_index}: AUC={auc:.3f}, threshold={optimal_threshold:.3f}, accuracy={accuracy:.3f}, flipped={flipped}")
        print(f"  Class distribution: {np.bincount(y)}")
        print(f"  Prediction distribution: {np.bincount(predictions)}")

    return accuracy


def get_multi_feature_auc_from_models(ml_model_dir, model_type, window_length,
                                      step_interval_min, verbose=False):
    """
    Get AUC value from trained multi-feature models by combining all predictions

    Parameters:
    - ml_model_dir: Directory containing trained models
    - model_type: Type of model ("rf" for Random Forest)
    - window_length: Window length in minutes
    - step_interval_min: Step interval in minutes
    - verbose: Print debug information

    Returns:
    - auc: AUC value calculated from all predictions
    """

    # Load predictions from all folds
    predictions_path = os.path.join(ml_model_dir, model_type,
                                    f"predictions_{model_type}_w{window_length}min_s{step_interval_min}min.csv")

    if not os.path.exists(predictions_path):
        print(f"Warning: Predictions file not found: {predictions_path}")
        return 0.0

    df_pred = pd.read_csv(predictions_path)

    if len(df_pred) == 0:
        print(f"Warning: No predictions found for window length {window_length}")
        return 0.0

    # Calculate AUC using all predictions
    auc = roc_auc_score(df_pred['true_label'], df_pred['predicted_prob'])

    if verbose:
        print(f"  Found {len(df_pred)} total predictions")

    return auc


def get_multi_feature_accuracy_from_models(ml_model_dir, model_type, window_length,
                                           step_interval_min, verbose=False):
    """
    Get accuracy value from trained multi-feature models by combining all predictions

    Parameters:
    - ml_model_dir: Directory containing trained models
    - model_type: Type of model ("rf" for Random Forest)
    - window_length: Window length in minutes
    - step_interval_min: Step interval in minutes
    - verbose: Print debug information

    Returns:
    - accuracy: Accuracy value calculated from all predictions
    """

    # Load predictions from all folds
    predictions_path = os.path.join(ml_model_dir, model_type,
                                    f"predictions_{model_type}_w{window_length}min_s{step_interval_min}min.csv")

    if not os.path.exists(predictions_path):
        print(f"Warning: Predictions file not found: {predictions_path}")
        return 0.0

    df_pred = pd.read_csv(predictions_path)

    if len(df_pred) == 0:
        print(f"Warning: No predictions found for window length {window_length}")
        return 0.0

    # Calculate accuracy using all predictions with 0.5 as threshold
    predictions = (df_pred['predicted_prob'] >= 0.5).astype(int)
    accuracy = accuracy_score(df_pred['true_label'], predictions)

    if verbose:
        print(f"  Found {len(df_pred)} total predictions")

    return accuracy


def load_feature_data(input_dir, window_length, step_interval_min):
    """
    Load feature data for a specific window length

    Parameters:
    - input_dir: Directory containing sliding window feature data
    - window_length: Window length in minutes
    - step_interval_min: Step interval in minutes

    Returns:
    - X: Feature matrix
    - y: Labels (1 for normal, 0 for blind)
    - groups: Group assignments for each sample
    """

    X = []
    y = []
    groups = []

    window_str = f"W{int(window_length)}min"
    step_str = f"S{step_interval_min:.2f}min"
    mouse_id = 0

    for fname in os.listdir(input_dir):
        if not fname.endswith(".txt"):
            continue

        # Check if both window length and step interval match
        if window_str not in fname or step_str not in fname:
            continue

        parts = fname.split('_')
        if len(parts) < 5:
            continue

        # Determine label: 1 for normal, 0 for blind
        label = 1 if parts[2].lower() == 'normal' else 0

        # Load features
        try:
            data = np.loadtxt(os.path.join(input_dir, fname))
            if len(data.shape) == 1:
                data = data.reshape(1, -1)

            # Append to datasets
            X.append(data)
            y.extend([label] * len(data))
            groups.extend([mouse_id] * len(data))

            mouse_id += 1

        except Exception as e:
            print(f"Error loading {fname}: {e}")
            continue

    if len(X) == 0:
        return np.array([]), np.array([]), np.array([])

    X = np.vstack(X)
    y = np.array(y)
    groups = np.array(groups)

    return X, y, groups


def create_auc_bar_plot(window_lengths, single_aucs, multi_aucs,
                        single_stds, multi_stds, single_feature_index):
    """
    Create bar plot comparing single-feature and multi-feature AUC values

    Parameters:
    - window_lengths: List of window lengths
    - single_aucs: List of single-feature AUC values
    - multi_aucs: List of multi-feature AUC values
    - single_stds: List of single-feature AUC standard deviations
    - multi_stds: List of multi-feature AUC standard deviations
    - single_feature_index: Index of the single feature used
    """

    # Feature names for legend
    feature_names = [
        "Time in dark (%)", "Light distance (cm)", "Dark distance (cm)",
        "Light→Dark crossings", "Immobile time light (s)", "Immobile time dark (s)",
        "Light avg speed (cm/s)", "Latency to dark SD", "Latency to light SD"
    ]

    single_feature_name = feature_names[single_feature_index]

    # Set up the plot
    plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

    x = np.arange(len(window_lengths))

    # Create bars (using colors similar to draw_boxplot.py Set2 palette)
    bars1 = plt.bar(x - BAR_WIDTH / 2, single_aucs, BAR_WIDTH,
                    label=f'Single-behavior feature assessment',
                    color='#2c3e50', alpha=0.8, edgecolor='black', linewidth=1)  # Orange from Set2

    bars2 = plt.bar(x + BAR_WIDTH / 2, multi_aucs, BAR_WIDTH,
                    label=f'Multi-behavior feature assessment',
                    color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1)  # Teal from Set2

    # Add error bars (set to zero since we have single AUC values)
    # plt.errorbar(x - BAR_WIDTH/2, single_aucs, yerr=single_stds,
    #             fmt='none', color='black', capsize=3, linewidth=1.5)
    # plt.errorbar(x + BAR_WIDTH/2, multi_aucs, yerr=multi_stds,
    #             fmt='none', color='black', capsize=3, linewidth=1.5)

    # Customize the plot
    plt.xlabel('Window length (min)', fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel('AUC', fontsize=AXIS_LABEL_FONTSIZE)
    plt.title('Classification performance comparison', fontsize=TITLE_FONTSIZE, y=1.05)

    # Set x-axis
    plt.xticks(x, [str(w) for w in window_lengths], fontsize=TICK_LABEL_FONTSIZE)
    plt.yticks(fontsize=TICK_LABEL_FONTSIZE)

    # Set y-axis limits to better show differences
    all_aucs = single_aucs + multi_aucs
    y_min = max(0.5, min(all_aucs) - 0.05)
    y_max = min(1.0, max(all_aucs) + 0.05)
    plt.ylim(y_min, y_max)

    # Add horizontal line at AUC = 0.5 (random chance)
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1)

    # Customize appearance
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_linewidth(AXIS_LINEWIDTH)
    plt.gca().spines['bottom'].set_linewidth(AXIS_LINEWIDTH)

    # Add legend (moved up)
    plt.legend(bbox_to_anchor=(1.0, 1.08), loc='upper right', frameon=False, fontsize=LEGEND_FONTSIZE)

    # Add AUC values on top of bars
    for i, (single_auc, multi_auc) in enumerate(zip(single_aucs, multi_aucs)):
        # Add text on single-feature bars
        plt.text(i - BAR_WIDTH / 2 - 0.1, single_auc + 0.003, f'{single_auc:.3f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
        # Add text on multi-feature bars
        plt.text(i + BAR_WIDTH / 2, multi_auc + 0.003, f'{multi_auc:.3f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add grid
    # plt.grid(True, axis='y', linestyle='--', alpha=0.3)

    # Tight layout and save
    plt.tight_layout()
    plt.savefig('Fig. 2(a)_ori.tif', dpi=300, bbox_inches='tight')
    plt.show()

    # Print summary
    print("\n=== AUC Summary ===")
    print("Window\tSingle-feature AUC\tMulti-feature AUC\tImprovement")
    print("-" * 55)
    for i, w in enumerate(window_lengths):
        improvement = multi_aucs[i] - single_aucs[i]
        print(f"{w} min\t{single_aucs[i]:.3f}\t\t{multi_aucs[i]:.3f}\t\t{improvement:+.3f}")

    print(f"\nFeature used for single-feature analysis: {single_feature_name}")