import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score

# Font size configuration
TITLE_FONTSIZE = 20
AXIS_LABEL_FONTSIZE = 18
TICK_LABEL_FONTSIZE = 14
LEGEND_FONTSIZE = 15

# Figure size configuration
FIGURE_WIDTH = 15
FIGURE_HEIGHT = 7

# Line and marker configuration
MODEL_LINE_WIDTH = 3
MARKER_SIZE = 8
MARKER_EDGE_WIDTH = 2
REFERENCE_LINE_WIDTH = 1.0
AXIS_LINEWIDTH = 1.5
BAR_EDGE_WIDTH = 1.5

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']

# Model configurations
MODEL_CONFIG = {
    'rf': {'name': 'Random Forest', 'color': '#2ecc71', 'marker': 'o', 'linestyle': '-'},
    'svm': {'name': 'SVM', 'color': '#3498db', 'marker': 's', 'linestyle': '-'},
    'xgb': {'name': 'XGBoost', 'color': '#e74c3c', 'marker': '^', 'linestyle': '-'},
    'logistic': {'name': 'Logistic Regression', 'color': '#f39c12', 'marker': 'D', 'linestyle': '-'},
    'mlp': {'name': 'MLP', 'color': '#9b59b6', 'marker': 'v', 'linestyle': '-'}
}


def compare_models_performance(model_dir, max_window_length=10, step_interval_min=0.5,
                               if_remove_features=False, models=None, verbose=False):
    """
    Compare performance across multiple models and window lengths

    Parameters:
    - model_dir: Directory containing trained models
    - max_window_length: Maximum window length to analyze
    - step_interval_min: Step interval in minutes
    - if_remove_features: Whether to use optimized feature models
    - models: List of model types to compare (default: all available)
    - verbose: Print debug information

    Returns:
    - Dictionary containing performance data for all models
    """

    if models is None:
        models = ['rf', 'svm', 'xgb', 'logistic', 'mlp']

    # Check which models are actually available
    available_models = []
    feature_suffix = "(opt_features)" if if_remove_features else ""

    for model in models:
        model_subdir = f"{model}{feature_suffix}"
        model_path = os.path.join(model_dir, model_subdir)
        if os.path.exists(model_path):
            available_models.append(model)
        elif verbose:
            print(f"Warning: Model directory not found: {model_path}")

    if not available_models:
        raise ValueError("No model directories found!")

    print(f"Comparing {len(available_models)} models: {available_models}")
    print(f"Using {'optimized features' if if_remove_features else 'all features'}")
    print(f"Window lengths: 1-{max_window_length} minutes")
    print("=" * 60)

    window_lengths = list(range(1, max_window_length + 1))
    results = {}

    # Collect performance data for each model
    for model_type in available_models:
        print(f"\nProcessing {model_type.upper()}...")
        model_aucs = []
        model_accs = []

        for window_length in window_lengths:
            auc, acc = get_model_performance(
                model_dir, model_type, window_length, step_interval_min,
                if_remove_features, verbose
            )
            model_aucs.append(auc)
            model_accs.append(acc)

            if verbose:
                print(f"  Window {window_length}min: AUC={auc:.3f}, ACC={acc:.3f}")

        results[model_type] = {
            'aucs': model_aucs,
            'accs': model_accs,
            'avg_auc': np.mean(model_aucs),
            'avg_acc': np.mean(model_accs)
        }

    # Create comparison plots
    create_multi_model_plots(window_lengths, results, if_remove_features)

    # Print summary
    print_multi_model_summary(window_lengths, results, available_models)

    return {
        'window_lengths': window_lengths,
        'models': available_models,
        'results': results,
        'if_remove_features': if_remove_features
    }


def get_model_performance(model_dir, model_type, window_length, step_interval_min,
                          if_remove_features=False, verbose=False):
    """
    Get AUC and accuracy for a specific model configuration

    Parameters:
    - model_dir: Base model directory
    - model_type: Type of model
    - window_length: Window length in minutes
    - step_interval_min: Step interval in minutes
    - if_remove_features: Whether to use optimized feature models
    - verbose: Print debug information

    Returns:
    - auc: AUC value
    - accuracy: Accuracy value
    """

    # Determine subdirectory
    feature_suffix = "(opt_features)" if if_remove_features else ""
    subdir = f"{model_type}{feature_suffix}"

    # Load predictions
    predictions_path = os.path.join(model_dir, subdir,
                                    f"predictions_{model_type}_w{window_length}min_s{step_interval_min}min.csv")

    if not os.path.exists(predictions_path):
        if verbose:
            print(f"Warning: Predictions file not found: {predictions_path}")
        return 0.0, 0.0

    try:
        df_pred = pd.read_csv(predictions_path)

        if len(df_pred) == 0:
            if verbose:
                print(f"Warning: No predictions found for {model_type} window length {window_length}")
            return 0.0, 0.0

        # Calculate overall AUC using all predictions
        auc = roc_auc_score(df_pred['true_label'], df_pred['predicted_prob'])

        # Calculate overall accuracy using 0.5 threshold
        predictions = (df_pred['predicted_prob'] >= 0.5).astype(int)
        accuracy = accuracy_score(df_pred['true_label'], predictions)

        return auc, accuracy

    except Exception as e:
        if verbose:
            print(f"Error loading predictions from {predictions_path}: {e}")
        return 0.0, 0.0


def create_multi_model_plots(window_lengths, results, if_remove_features):
    """
    Create line plots comparing AUC and Accuracy across models and window lengths
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

    # Sort models by average AUC (descending)
    sorted_models = sorted(results.keys(), key=lambda x: results[x]['avg_auc'], reverse=True)

    # Plot AUC comparison
    for model_type in sorted_models:
        data = results[model_type]
        config = MODEL_CONFIG.get(model_type,
                                  {'name': model_type.upper(), 'color': 'black', 'marker': 'o', 'linestyle': '-'})
        ax1.plot(window_lengths, data['aucs'],
                 label=f"{config['name']} (avg: {data['avg_auc']:.3f})",
                 color=config['color'], marker=config['marker'],
                 linestyle=config['linestyle'], linewidth=MODEL_LINE_WIDTH, markersize=MARKER_SIZE,
                 markerfacecolor='white', markeredgewidth=MARKER_EDGE_WIDTH)

    ax1.set_xlabel('Window length (min)', fontsize=AXIS_LABEL_FONTSIZE)
    ax1.set_ylabel('AUC', fontsize=AXIS_LABEL_FONTSIZE)
    ax1.set_title('', fontsize=TITLE_FONTSIZE)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=REFERENCE_LINE_WIDTH)

    # Plot Accuracy comparison
    for model_type in sorted_models:
        data = results[model_type]
        config = MODEL_CONFIG.get(model_type,
                                  {'name': model_type.upper(), 'color': 'black', 'marker': 'o', 'linestyle': '-'})
        ax2.plot(window_lengths, data['accs'],
                 label=f"{config['name']} (avg: {data['avg_acc']:.3f})",
                 color=config['color'], marker=config['marker'],
                 linestyle=config['linestyle'], linewidth=MODEL_LINE_WIDTH, markersize=MARKER_SIZE,
                 markerfacecolor='white', markeredgewidth=MARKER_EDGE_WIDTH)

    ax2.set_xlabel('Window length (min)', fontsize=AXIS_LABEL_FONTSIZE)
    ax2.set_ylabel('Accuracy', fontsize=AXIS_LABEL_FONTSIZE)
    ax2.set_title('', fontsize=TITLE_FONTSIZE)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=REFERENCE_LINE_WIDTH)

    # Customize axes
    for ax in [ax1, ax2]:
        ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(AXIS_LINEWIDTH)
        ax.spines['bottom'].set_linewidth(AXIS_LINEWIDTH)
        ax.set_xticks(window_lengths)

        # Set unified y-axis limits
        ax.set_ylim(0.6, 0.85)

    # Add legends
    ax1.legend(loc='upper left', frameon=False, fontsize=LEGEND_FONTSIZE - 2, bbox_to_anchor=(0, 1.04))
    ax2.legend(loc='upper left', frameon=False, fontsize=LEGEND_FONTSIZE - 2, bbox_to_anchor=(0, 1.04))

    plt.tight_layout()

    # Save figure
    filename = f"Multi_Model_Comparison_{'opt_features' if if_remove_features else 'all_features'}.tif"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


def create_summary_bar_plot(results, if_remove_features):
    """
    Create a summary bar plot showing average performance across all window lengths
    """

    plt.figure(figsize=(12, 6))

    models = list(results.keys())
    avg_aucs = [results[model]['avg_auc'] for model in models]
    avg_accs = [results[model]['avg_acc'] for model in models]

    x = np.arange(len(models))
    bar_width = 0.35

    # Create bars
    bars1 = plt.bar(x - bar_width / 2, avg_aucs, bar_width,
                    label='Average AUC', color='#3498db', alpha=0.8,
                    edgecolor='black', linewidth=BAR_EDGE_WIDTH)
    bars2 = plt.bar(x + bar_width / 2, avg_accs, bar_width,
                    label='Average Accuracy', color='#e74c3c', alpha=0.8,
                    edgecolor='black', linewidth=BAR_EDGE_WIDTH)

    # Customize the plot
    plt.xlabel('Model', fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel('Performance', fontsize=AXIS_LABEL_FONTSIZE)
    feature_type = "optimized features" if if_remove_features else "all features"
    plt.title(f'Average Performance Comparison ({feature_type})', fontsize=TITLE_FONTSIZE)

    # Set x-axis
    model_names = [MODEL_CONFIG.get(model, {'name': model.upper()})['name'] for model in models]
    plt.xticks(x, model_names, fontsize=TICK_LABEL_FONTSIZE)
    plt.yticks(fontsize=TICK_LABEL_FONTSIZE)

    # Add horizontal line at 0.5
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=REFERENCE_LINE_WIDTH)

    # Customize appearance
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_linewidth(AXIS_LINEWIDTH)
    plt.gca().spines['bottom'].set_linewidth(AXIS_LINEWIDTH)

    # Add values on top of bars
    for i, (auc, acc) in enumerate(zip(avg_aucs, avg_accs)):
        plt.text(i - bar_width / 2, auc + 0.005, f'{auc:.3f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
        plt.text(i + bar_width / 2, acc + 0.005, f'{acc:.3f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add legend
    plt.legend(frameon=False, fontsize=LEGEND_FONTSIZE)

    plt.tight_layout()
    filename = f"Model_Average_Performance_{'opt_features' if if_remove_features else 'all_features'}.tif"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


def print_multi_model_summary(window_lengths, results, models):
    """
    Print detailed performance comparison summary across all models
    """

    print("\n" + "=" * 100)
    print("MULTI-MODEL PERFORMANCE COMPARISON SUMMARY")
    print("=" * 100)

    # Performance by window length
    print("\n=== AUC by Window Length ===")
    header = "Window\t" + "\t".join([f"{model.upper():<8}" for model in models])
    print(header)
    print("-" * len(header.expandtabs()))

    for i, w in enumerate(window_lengths):
        row = f"{w} min\t"
        for model in models:
            row += f"{results[model]['aucs'][i]:.3f}\t\t"
        print(row.rstrip())

    print("\n=== Accuracy by Window Length ===")
    print(header)
    print("-" * len(header.expandtabs()))

    for i, w in enumerate(window_lengths):
        row = f"{w} min\t"
        for model in models:
            row += f"{results[model]['accs'][i]:.3f}\t\t"
        print(row.rstrip())

    # Performance ranges for each model
    print(f"\n=== Performance Ranges (Min-Max) ===")
    print("Model\t\t\tAUC Range\t\tAccuracy Range")
    print("-" * 70)

    # Sort models by average AUC for ranking
    model_ranking = sorted(models, key=lambda x: results[x]['avg_auc'], reverse=True)

    for model in model_ranking:
        model_name = MODEL_CONFIG.get(model, {'name': model.upper()})['name']
        auc_min = min(results[model]['aucs'])
        auc_max = max(results[model]['aucs'])
        acc_min = min(results[model]['accs'])
        acc_max = max(results[model]['accs'])
        print(f"{model_name:<20}\t{auc_min:.3f}-{auc_max:.3f}\t\t{acc_min:.3f}-{acc_max:.3f}")

    print("=" * 100)