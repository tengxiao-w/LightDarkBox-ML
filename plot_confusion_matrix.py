import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns

# Font size configuration (matching draw_roc.py style)
TITLE_FONTSIZE = 18
AXIS_LABEL_FONTSIZE = 16
TICK_LABEL_FONTSIZE = 14
LEGEND_FONTSIZE = 12
AXIS_LINEWIDTH = 1.5

FIGURE_WIDTH = 6
FIGURE_HEIGHT = 5

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']

# Model configurations
MODEL_CONFIG = {
    'rf': {'name': 'Random Forest'},
    'svm': {'name': 'SVM'},
    'xgb': {'name': 'XGBoost'},
    'logistic': {'name': 'Logistic Regression'},
    'mlp': {'name': 'MLP'}
}


def plot_confusion_matrix(ml_model_dir, model_type="rf", window_length=5,
                          step_interval_min=0.5, threshold=0.5,
                          if_remove_features=False, verbose=False):
    """
    Plot confusion matrix for a specific model and window length

    Parameters:
    - ml_model_dir: Directory containing trained ML models
    - model_type: Type of ML model ("rf", "svm", "xgb", "logistic", "mlp")
    - window_length: Window length in minutes
    - step_interval_min: Step interval in minutes
    - threshold: Classification threshold (default: 0.5)
    - if_remove_features: Whether to use optimized feature models
    - verbose: Print debug information
    """

    print(f"Plotting confusion matrix for {model_type.upper()} model")
    print(f"Window length: {window_length} min")
    print(f"Step interval: {step_interval_min} min")
    print(f"Classification threshold: {threshold}")
    print(f"Using {'optimized features' if if_remove_features else 'all features'}")
    print("=" * 60)

    # Load predictions
    y_true, y_pred, y_prob = load_model_predictions(
        ml_model_dir, model_type, window_length, step_interval_min,
        threshold, if_remove_features, verbose
    )

    if y_true is None or len(y_true) == 0:
        print("Error: Could not load predictions")
        return

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    # Calculate sensitivity and specificity
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate

    # Print classification metrics
    print(f"\n=== Classification Metrics ===")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-score: {f1:.3f}")
    print(f"Sensitivity: {sensitivity:.3f}")
    print(f"Specificity: {specificity:.3f}")
    print(f"Total samples: {len(y_true)}")

    # Create confusion matrix plot
    create_confusion_matrix_plot(cm, model_type, window_length, accuracy,
                                 sensitivity, specificity, if_remove_features)

    return {
        'confusion_matrix': cm,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'total_samples': len(y_true)
    }


def load_model_predictions(ml_model_dir, model_type, window_length, step_interval_min,
                           threshold, if_remove_features=False, verbose=False):
    """
    Load model predictions from CSV file

    Parameters:
    - ml_model_dir: Directory containing trained models
    - model_type: Type of model
    - window_length: Window length in minutes
    - step_interval_min: Step interval in minutes
    - threshold: Classification threshold
    - if_remove_features: Whether to use optimized feature models
    - verbose: Print debug information

    Returns:
    - y_true: True labels
    - y_pred: Predicted labels
    - y_prob: Predicted probabilities
    """

    # Determine subdirectory
    feature_suffix = "(opt_features)" if if_remove_features else ""
    subdir = f"{model_type}{feature_suffix}"

    # Load predictions
    predictions_path = os.path.join(ml_model_dir, subdir,
                                    f"predictions_{model_type}_w{window_length}min_s{step_interval_min}min.csv")

    if not os.path.exists(predictions_path):
        print(f"Error: Predictions file not found: {predictions_path}")
        return None, None, None

    try:
        df_pred = pd.read_csv(predictions_path)

        if len(df_pred) == 0:
            print(f"Error: No predictions found for {model_type} window length {window_length}")
            return None, None, None

        # Extract true labels and predicted probabilities
        y_true = df_pred['true_label'].values
        y_prob = df_pred['predicted_prob'].values

        # Apply threshold to get predicted labels
        y_pred = (y_prob >= threshold).astype(int)

        if verbose:
            print(f"Loaded {len(y_true)} predictions")
            print(f"Class distribution - True: {np.bincount(y_true)}")
            print(f"Class distribution - Predicted: {np.bincount(y_pred)}")

        return y_true, y_pred, y_prob

    except Exception as e:
        print(f"Error loading predictions from {predictions_path}: {e}")
        return None, None, None


def create_confusion_matrix_plot(cm, model_type, window_length, accuracy,
                                 sensitivity, specificity, if_remove_features):
    """
    Create and save confusion matrix plot

    Parameters:
    - cm: Confusion matrix
    - model_type: Type of model
    - window_length: Window length in minutes
    - accuracy: Accuracy score
    - sensitivity: Sensitivity score
    - specificity: Specificity score
    - if_remove_features: Whether using optimized features
    """

    plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

    # Define class labels
    class_labels = ['Blind', 'Sighted']

    # Calculate percentages for each cell
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create heatmap using seaborn
    # Use a yellow-orange-red colormap
    sns.heatmap(cm, annot=False, fmt='d', cmap='YlOrRd',
                square=True, linewidths=2, linecolor='white',
                xticklabels=class_labels, yticklabels=class_labels,
                cbar_kws={'shrink': 0.8})

    # Add custom annotations with only percentages
    for i in range(len(class_labels)):
        for j in range(len(class_labels)):
            plt.text(j + 0.5, i + 0.5, f'{cm_percentage[i, j]:.3f}',
                     ha='center', va='center', fontsize=14, fontweight='bold',
                     color='white' if cm[i, j] > cm.max() * 0.5 else 'black')

    # Get model name for title
    model_name = MODEL_CONFIG.get(model_type, {'name': model_type.upper()})['name']
    feature_type = "optimized features" if if_remove_features else "all features"

    # Customize labels and title
    plt.xlabel('Predicted label', fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel('True label', fontsize=AXIS_LABEL_FONTSIZE)
    plt.title(f'{model_name} confusion matrix\n(window length = {window_length} min)',
              fontsize=TITLE_FONTSIZE, pad=20)

    # Customize ticks
    plt.xticks(fontsize=TICK_LABEL_FONTSIZE)
    plt.yticks(fontsize=TICK_LABEL_FONTSIZE, rotation=0)

    # Add performance metrics as text (also convert to percentage format)
    metrics_text = f'Accuracy: {accuracy:.3f}\nSensitivity: {sensitivity:.3f}\nSpecificity: {specificity:.3f}'
    plt.text(1.05, -0.15, metrics_text, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='bottom', horizontalalignment='left',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))

    # Tight layout and save
    plt.tight_layout()

    # Generate filename
    feature_suffix = "_opt_features" if if_remove_features else "_all_features"
    filename = "Fig. 2(b)_ori.tif"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nConfusion matrix saved as: {filename}")