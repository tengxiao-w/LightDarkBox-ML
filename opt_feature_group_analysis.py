import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score

# Font size configuration (matching other modules)
TITLE_FONTSIZE = 20
AXIS_LABEL_FONTSIZE = 16
TICK_LABEL_FONTSIZE = 14
LEGEND_FONTSIZE = 14
AXIS_LINEWIDTH = 1.5
BAR_WIDTH = 0.35

FIGURE_WIDTH = 15
FIGURE_HEIGHT = 5

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']


def compare_performance(model_dir, max_window_length=10, step_interval_min=0.5,
                              model_type="rf", verbose=False):
    """
    Compare performance between original models and optimized feature models

    Parameters:
    - model_dir: Directory containing trained models
    - max_window_length: Maximum window length to analyze
    - step_interval_min: Step interval in minutes
    - model_type: Type of model ("rf" for Random Forest)
    - verbose: Print debug information
    """

    window_lengths = list(range(1, max_window_length + 1))
    original_aucs = []
    optimized_aucs = []
    original_accs = []
    optimized_accs = []

    print(f"Comparing model performance for window lengths 1-{max_window_length} minutes...")
    print(f"Original models directory: {os.path.join(model_dir, model_type)}")
    print(f"Optimized models directory: {os.path.join(model_dir, f'{model_type}(opt_features)')}")

    for window_length in window_lengths:
        print(f"\nProcessing window length = {window_length} min")

        # Get original model performance
        orig_auc, orig_acc = get_model_performance(
            model_dir, model_type, window_length, step_interval_min, optimized=False, verbose=verbose
        )
        original_aucs.append(orig_auc)
        original_accs.append(orig_acc)

        # Get optimized model performance
        opt_auc, opt_acc = get_model_performance(
            model_dir, model_type, window_length, step_interval_min, optimized=True, verbose=verbose
        )
        optimized_aucs.append(opt_auc)
        optimized_accs.append(opt_acc)

        if verbose:
            print(f"  Original - AUC: {orig_auc:.3f}, ACC: {orig_acc:.3f}")
            print(f"  Optimized - AUC: {opt_auc:.3f}, ACC: {opt_acc:.3f}")
            print(f"  Improvement - AUC: {opt_auc - orig_auc:+.3f}, ACC: {opt_acc - orig_acc:+.3f}")

    # Create comparison plots
    create_comparison_plots(window_lengths, original_aucs, optimized_aucs,
                            original_accs, optimized_accs, model_type)

    # Print summary
    print_performance_summary(window_lengths, original_aucs, optimized_aucs,
                              original_accs, optimized_accs)

    return {
        'window_lengths': window_lengths,
        'original_aucs': original_aucs,
        'optimized_aucs': optimized_aucs,
        'original_accs': original_accs,
        'optimized_accs': optimized_accs
    }


def get_model_performance(model_dir, model_type, window_length, step_interval_min,
                          optimized=False, verbose=False):
    """
    Get AUC and accuracy for a specific model configuration

    Parameters:
    - model_dir: Base model directory
    - model_type: Type of model
    - window_length: Window length in minutes
    - step_interval_min: Step interval in minutes
    - optimized: Whether to use optimized feature models
    - verbose: Print debug information

    Returns:
    - auc: AUC value
    - accuracy: Accuracy value
    """

    # Determine subdirectory
    if optimized:
        subdir = f"{model_type}(opt_features)"
    else:
        subdir = model_type

    # Load predictions
    predictions_path = os.path.join(model_dir, subdir,
                                    f"predictions_{model_type}_w{window_length}min_s{step_interval_min}min.csv")

    if not os.path.exists(predictions_path):
        print(f"Warning: Predictions file not found: {predictions_path}")
        return 0.0, 0.0

    try:
        df_pred = pd.read_csv(predictions_path)

        if len(df_pred) == 0:
            print(f"Warning: No predictions found for window length {window_length}")
            return 0.0, 0.0

        # Calculate overall AUC using all predictions
        auc = roc_auc_score(df_pred['true_label'], df_pred['predicted_prob'])

        # Calculate overall accuracy using 0.5 threshold
        predictions = (df_pred['predicted_prob'] >= 0.5).astype(int)
        accuracy = accuracy_score(df_pred['true_label'], predictions)

        if verbose:
            print(f"    Loaded {len(df_pred)} predictions from {subdir}")

        return auc, accuracy

    except Exception as e:
        print(f"Error loading predictions from {predictions_path}: {e}")
        return 0.0, 0.0


def create_comparison_plots(window_lengths, original_aucs, optimized_aucs,
                            original_accs, optimized_accs, model_type):
    """
    Create comparison bar plots with both AUC and Accuracy in one plot (4 bars per window)
    Order: ACC first, then AUC
    """

    plt.figure(figsize=(FIGURE_WIDTH + 2, FIGURE_HEIGHT))

    x = np.arange(len(window_lengths))
    bar_width = 0.22  # Narrower bars to fit 4 bars per group

    # Create bars - 4 bars per window (ACC first, then AUC)
    bars1 = plt.bar(x - 1.5 * bar_width, original_accs, bar_width,
                    label=f'Accuracy (All features)',
                    color='#808080', alpha=0.5, edgecolor='black', linewidth=1.5)  # Gray, solid

    bars2 = plt.bar(x - 0.5 * bar_width, optimized_accs, bar_width,
                    label=f'Accuracy (Optimized subset)',
                    color='#2980b9', alpha=0.8, edgecolor='black', linewidth=1.5)  # Blue, hatched

    bars3 = plt.bar(x + 0.5 * bar_width, original_aucs, bar_width,
                    label=f'AUC (All features)',
                    color='#808080', alpha=0.5, edgecolor='black', linewidth=1.5,
                    hatch='///')  # Gray, hatched

    bars4 = plt.bar(x + 1.5 * bar_width, optimized_aucs, bar_width,
                    label=f'AUC (Optimized subset)',
                    color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)  # Red, solid

    # Customize the plot
    plt.xlabel('Window length (min)', fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel('Performance', fontsize=AXIS_LABEL_FONTSIZE)
    plt.title(f'{model_type.upper()} performance comparison: all features vs. optimized subset', fontsize=TITLE_FONTSIZE, y=1.15)

    # Set x-axis
    plt.xticks(x, [str(w) for w in window_lengths], fontsize=TICK_LABEL_FONTSIZE)
    plt.yticks(fontsize=TICK_LABEL_FONTSIZE)

    # Reduce white space on both sides
    plt.xlim(-0.5, len(window_lengths) - 0.5)

    # Set y-axis limits
    all_values = original_aucs + optimized_aucs + original_accs + optimized_accs
    y_min = max(0.5, min(all_values) - 0.05)
    y_max = min(1.0, max(all_values) + 0.05)
    plt.ylim(y_min, y_max)

    # Add horizontal line at 0.5 (random chance)
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1)

    # Customize appearance
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_linewidth(AXIS_LINEWIDTH)
    plt.gca().spines['bottom'].set_linewidth(AXIS_LINEWIDTH)

    # Add legend
    plt.legend(bbox_to_anchor=(0.5, 1.15), loc='upper center', frameon=False, fontsize=LEGEND_FONTSIZE, ncol=4)

    # Add values on top of bars (smaller font to avoid crowding)
    for i, (orig_acc, opt_acc, orig_auc, opt_auc) in enumerate(
            zip(original_accs, optimized_accs, original_aucs, optimized_aucs)):
        plt.text(i - 1.5 * bar_width, orig_acc + 0.003, f'{orig_acc:.3f}',
                 ha='center', va='bottom', fontsize=8, fontweight='bold')
        plt.text(i - 0.5 * bar_width, opt_acc + 0.003, f'{opt_acc:.3f}',
                 ha='center', va='bottom', fontsize=8, fontweight='bold')
        plt.text(i + 0.5 * bar_width, orig_auc + 0.003, f'{orig_auc:.3f}',
                 ha='center', va='bottom', fontsize=8, fontweight='bold')
        plt.text(i + 1.5 * bar_width, opt_auc + 0.003, f'{opt_auc:.3f}',
                 ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'Fig. 3(e)_{model_type.upper()}_ori.tif', dpi=300, bbox_inches='tight')
    plt.show()


def print_performance_summary(window_lengths, original_aucs, optimized_aucs,
                              original_accs, optimized_accs):
    """
    Print detailed performance comparison summary
    """

    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("=" * 80)

    print("\n=== AUC Comparison ===")
    print("Window\tOriginal\tOptimized\tImprovement")
    print("-" * 50)
    for i, w in enumerate(window_lengths):
        improvement = optimized_aucs[i] - original_aucs[i]
        print(f"{w} min\t{original_aucs[i]:.3f}\t\t{optimized_aucs[i]:.3f}\t\t{improvement:+.3f}")

    print("\n=== Accuracy Comparison ===")
    print("Window\tOriginal\tOptimized\tImprovement")
    print("-" * 50)
    for i, w in enumerate(window_lengths):
        improvement = optimized_accs[i] - original_accs[i]
        print(f"{w} min\t{original_accs[i]:.3f}\t\t{optimized_accs[i]:.3f}\t\t{improvement:+.3f}")

    # Calculate overall statistics
    avg_auc_improvement = np.mean([optimized_aucs[i] - original_aucs[i] for i in range(len(window_lengths))])
    avg_acc_improvement = np.mean([optimized_accs[i] - original_accs[i] for i in range(len(window_lengths))])

    print(f"\n=== Overall Statistics ===")
    print(f"Average AUC improvement: {avg_auc_improvement:+.3f}")
    print(f"Average Accuracy improvement: {avg_acc_improvement:+.3f}")

    # Find best performing window lengths
    best_orig_auc_idx = np.argmax(original_aucs)
    best_opt_auc_idx = np.argmax(optimized_aucs)
    best_orig_acc_idx = np.argmax(original_accs)
    best_opt_acc_idx = np.argmax(optimized_accs)

    print(f"\n=== Best Performance ===")
    print(f"Best original AUC: {original_aucs[best_orig_auc_idx]:.3f} at {window_lengths[best_orig_auc_idx]} min")
    print(f"Best optimized AUC: {optimized_aucs[best_opt_auc_idx]:.3f} at {window_lengths[best_opt_auc_idx]} min")
    print(f"Best original Accuracy: {original_accs[best_orig_acc_idx]:.3f} at {window_lengths[best_orig_acc_idx]} min")
    print(f"Best optimized Accuracy: {optimized_accs[best_opt_acc_idx]:.3f} at {window_lengths[best_opt_acc_idx]} min")

    print("=" * 80)