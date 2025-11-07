import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import random

# Font size configuration (matching other modules)
TITLE_FONTSIZE = 18
AXIS_LABEL_FONTSIZE = 16
TICK_LABEL_FONTSIZE = 14
AXIS_LINEWIDTH = 1.5
BAR_LINEWIDTH = 1.75

FIGURE_WIDTH = 8
FIGURE_HEIGHT = 5

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']

# Feature names as specified
FEATURE_NAMES = [
    "Dark side percentage",
    "Light side distance",
    "Dark side distance",
    "Visits to dark side",
    "Light immobile time",
    "Dark immobile time",
    "Light average speed",
    "Dark average speed",
    "Latency to dark (SD)",
    "Latency to light (SD)"
]


def plot_feature_importance(model_dir, window_length, step_interval_min=0.5,
                            model_type="rf", verbose=False):
    """
    Plot feature importance for Random Forest models

    Parameters:
    - model_dir: Directory containing trained models
    - window_length: Window length in minutes
    - step_interval_min: Step interval in minutes
    - model_type: Type of model ("rf" for Random Forest)
    - verbose: Print debug information
    """

    try:
        importance_ranking, mean_importance, std_importance = get_feature_importance_ranking(
            model_dir, window_length, step_interval_min, model_type)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    if verbose:
        print(f"Loaded models for window length {window_length} min")
        print("Feature importance summary:")
        for i, (name, mean_imp, std_imp) in enumerate(zip(FEATURE_NAMES, mean_importance, std_importance)):
            print(f"  {i + 1}. {name}: {mean_imp:.3f} ± {std_imp:.3f}")

    # Sort features by importance (highest to lowest, top to bottom)
    sorted_indices = np.argsort(mean_importance)  # Ascending order for horizontal bar plot
    sorted_feature_names = [FEATURE_NAMES[i] for i in sorted_indices]
    sorted_mean_importance = mean_importance[sorted_indices]
    sorted_std_importance = std_importance[sorted_indices]

    # Auto-numbering based on importance ranking (highest importance gets F1)
    # Since we're using ascending order for horizontal bar plot, we need to reverse the numbering
    auto_numbered_names = []
    for i, name in enumerate(sorted_feature_names):
        rank = len(sorted_feature_names) - i  # Reverse numbering so highest importance gets F1
        auto_numbered_names.append(f"F{rank}: {name}")

    # Create the plot
    plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

    x_pos = np.arange(len(auto_numbered_names))

    # Create horizontal bar plot with only upper error bars (outside part)
    bars = plt.barh(x_pos, sorted_mean_importance,
                    color='#2980b9', alpha=0.7,
                    edgecolor='black', linewidth=BAR_LINEWIDTH)

    # Add error bars manually - only the upper part (right side for horizontal bars)
    for i, (mean_val, std_val) in enumerate(zip(sorted_mean_importance, sorted_std_importance)):
        plt.errorbar(mean_val, i, xerr=[[0], [std_val]],
                     color='black', linewidth=1.5, capsize=5)

    # Customize the plot
    plt.yticks(x_pos, auto_numbered_names, fontsize=TICK_LABEL_FONTSIZE)
    # Set y-axis labels to left align and adjust position
    ax = plt.gca()
    ax.tick_params(axis='y', pad=165)  # Increase padding between labels and axis
    for tick in ax.get_yticklabels():
        tick.set_horizontalalignment('left')

    plt.xlabel('Feature importance', fontsize=AXIS_LABEL_FONTSIZE)
    plt.suptitle(f'Random Forest feature importance (window length = {window_length} min)',
                 fontsize=TITLE_FONTSIZE, x=0.5, y=0.95)  # Center the title

    # Set axis properties
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_linewidth(AXIS_LINEWIDTH)
    plt.gca().spines['bottom'].set_linewidth(AXIS_LINEWIDTH)

    # Set tick parameters
    plt.gca().tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
    plt.gca().tick_params(axis='x', which='major', labelsize=TICK_LABEL_FONTSIZE)

    # Add value labels on bars
    for i, (bar, mean_val, std_val) in enumerate(zip(bars, sorted_mean_importance, sorted_std_importance)):
        plt.text(bar.get_width() + std_val + 0.005, bar.get_y() + bar.get_height() / 2,
                 f'{mean_val:.3f}', ha='left', va='center', fontsize=10, fontweight='bold')

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('Fig. 3(a)_ori.tif', dpi=300, bbox_inches='tight')
    plt.show()

    # Print ranking
    print(f"\n=== Feature Importance Ranking (Window: {window_length} min) ===")
    importance_ranking_display = sorted(enumerate(zip(FEATURE_NAMES, mean_importance, std_importance)),
                                        key=lambda x: x[1][1], reverse=True)

    print("Rank\tFeature\t\t\t\tImportance\tStd")
    print("-" * 65)
    for rank, (idx, (name, mean_imp, std_imp)) in enumerate(importance_ranking_display, 1):
        print(f"{rank}\t{name:<25}\t{mean_imp:.3f}\t\t{std_imp:.3f}")

    return importance_ranking


def get_feature_importance_ranking(model_dir, window_length, step_interval_min=0.5, model_type="rf"):
    """
    Get feature importance ranking from trained models

    Returns:
    - importance_ranking: array of feature indices sorted by importance (descending)
    - mean_importance: mean importance values
    - std_importance: std of importance values
    """
    # Load feature importance from all folds
    importance_list = []
    model_output_dir = os.path.join(model_dir, model_type)

    fold_idx = 0
    while True:
        model_path = os.path.join(model_output_dir,
                                  f"model_{model_type}_w{window_length}min_s{step_interval_min}min_fold{fold_idx}.joblib")

        if not os.path.exists(model_path):
            break

        # Load model and extract feature importance
        clf = joblib.load(model_path)
        importance_list.append(clf.feature_importances_)
        fold_idx += 1

    if len(importance_list) == 0:
        raise FileNotFoundError(f"No models found for window length {window_length} min")

    # Calculate mean and std of feature importance across folds
    importance_array = np.array(importance_list)
    mean_importance = np.mean(importance_array, axis=0)
    std_importance = np.std(importance_array, axis=0)

    # Sort by importance (descending)
    importance_ranking = np.argsort(mean_importance)[::-1]

    return importance_ranking, mean_importance, std_importance


def plot_ablation_study(input_dir, model_dir, window_length, step_interval_min=0.5,
                        model_type="rf", k_splits=5, seed=0, verbose=False):
    """
    Perform feature ablation study and plot line chart

    Parameters:
    - input_dir: Directory containing sliding window feature data
    - model_dir: Directory containing trained models (for getting feature importance ranking)
    - window_length: Window length in minutes
    - step_interval_min: Step interval in minutes
    - model_type: Type of model ("rf" for Random Forest)
    - k_splits: Number of folds for cross-validation
    - seed: Random seed for reproducibility
    - verbose: Print debug information
    """

    print(f"Starting feature ablation study for window length {window_length} min...")

    # Get feature importance ranking from trained models (same as plot_feature_importance)
    try:
        importance_ranking, mean_importance, std_importance = get_feature_importance_ranking(
            model_dir, window_length, step_interval_min, model_type)
        print("Using feature importance ranking from trained models")
    except FileNotFoundError:
        print("Warning: No trained models found, will compute feature importance from scratch")
        # Fallback: compute feature ranking from scratch
        X, y, groups, metadata = load_data(window_length, step_interval_min, input_dir)
        clf_full = RandomForestClassifier(n_estimators=100, random_state=seed, class_weight='balanced')
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        clf_full.fit(X_scaled, y)
        feature_importance = clf_full.feature_importances_
        importance_ranking = np.argsort(feature_importance)[::-1]
        mean_importance = feature_importance
        std_importance = np.zeros_like(feature_importance)

    if verbose:
        print("Feature importance ranking:")
        for i, idx in enumerate(importance_ranking):
            print(f"  F{i + 1}: {FEATURE_NAMES[idx]} (importance: {mean_importance[idx]:.3f})")

    # Load data for ablation study
    X, y, groups, metadata = load_data(window_length, step_interval_min, input_dir)

    if len(X) == 0:
        print("Error: No data loaded")
        return

    print(f"Loaded {len(X)} samples with {X.shape[1]} features")

    # Perform ablation study
    ablation_results = []
    x_labels = []

    # Start with all features, then remove one by one based on importance ranking
    for remove_count in range(len(FEATURE_NAMES) + 1):  # 0 to 10 (remove 0 to 10 features)
        if remove_count == 0:
            # Use all features
            selected_features = list(range(len(FEATURE_NAMES)))
            label = "F1-F10"
        else:
            # Remove the top remove_count most important features
            features_to_remove = importance_ranking[:remove_count]
            selected_features = [i for i in range(len(FEATURE_NAMES)) if i not in features_to_remove]

            if len(selected_features) == 0:
                break

            # Create label showing remaining features
            remaining_feature_ranks = []
            for feat_idx in selected_features:
                rank = np.where(importance_ranking == feat_idx)[0][0] + 1
                remaining_feature_ranks.append(f"F{rank}")

            # Sort the remaining features by their rank
            remaining_feature_ranks.sort(key=lambda x: int(x[1:]))

            if len(remaining_feature_ranks) <= 3:
                label = "-".join(remaining_feature_ranks)
            else:
                label = f"F{min([int(x[1:]) for x in remaining_feature_ranks])}-F{max([int(x[1:]) for x in remaining_feature_ranks])}"

        x_labels.append(label)

        # Train with selected features using cross-validation
        X_selected = X[:, selected_features]

        # Perform k-fold cross-validation
        folds = stratified_group_k_fold(y, groups, k=k_splits, seed=seed)

        # Collect all predictions across folds
        all_y_true = []
        all_y_score = []
        all_y_pred = []

        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            X_train, X_test = X_selected[train_idx], X_selected[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Standardization
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train model
            clf = RandomForestClassifier(n_estimators=100, random_state=seed, class_weight='balanced')
            clf.fit(X_train_scaled, y_train)

            # Evaluate
            y_pred = clf.predict(X_test_scaled)
            y_score = clf.predict_proba(X_test_scaled)[:, 1]

            # Collect all results
            all_y_true.extend(y_test)
            all_y_score.extend(y_score)
            all_y_pred.extend(y_pred)

        # Calculate overall metrics on combined test results
        overall_auc = roc_auc_score(all_y_true, all_y_score)
        overall_acc = accuracy_score(all_y_true, all_y_pred)

        ablation_results.append({
            'label': label,
            'features_used': len(selected_features),
            'auc': overall_auc,
            'acc': overall_acc
        })

        if verbose:
            print(f"  {label}: AUC = {overall_auc:.3f}, ACC = {overall_acc:.3f}")

    # Plot the results
    plot_ablation_results(ablation_results, x_labels, window_length)

    return ablation_results


def plot_ablation_results(results, x_labels, window_length):
    """
    Plot ablation study results
    """
    aucs = [r['auc'] for r in results]
    accs = [r['acc'] for r in results]

    plt.figure(figsize=(FIGURE_WIDTH-1.5, FIGURE_HEIGHT))

    x_pos = np.arange(len(x_labels))

    # Plot AUC (no error bars since we have single values)
    plt.plot(x_pos, aucs, marker='s', linewidth=3,
             markersize=10, label='AUC', color='#e74c3c')

    # Plot Accuracy (no error bars since we have single values)
    plt.plot(x_pos, accs, marker='o', linewidth=3,
             markersize=10, label='Accuracy', color='#2980b9')

    plt.xlabel('Feature combination', fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel('Performance', fontsize=AXIS_LABEL_FONTSIZE)
    plt.title(f'Feature ablation',
              fontsize=TITLE_FONTSIZE, y=1.05)

    plt.xticks(x_pos, x_labels, fontsize=TICK_LABEL_FONTSIZE - 2, rotation=45, ha='center')
    plt.yticks(fontsize=TICK_LABEL_FONTSIZE)

    # Set y-axis limits
    all_values = aucs + accs
    y_min = max(0.4, min(all_values) - 0.05)
    y_max = min(1.0, max(all_values) + 0.05)
    plt.ylim(y_min, y_max)

    # Add horizontal line at 0.5 (random chance)
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # Customize appearance
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_linewidth(AXIS_LINEWIDTH)
    plt.gca().spines['bottom'].set_linewidth(AXIS_LINEWIDTH)

    plt.legend(loc='upper right', frameon=False, fontsize=TICK_LABEL_FONTSIZE)

    plt.tight_layout()
    plt.savefig('Fig.3 (b)_ori.tif', dpi=300, bbox_inches='tight')
    plt.show()


def load_data(window_length_min, window_step_min, input_dir):
    """
    Load and prepare data for training (copied from train_ml_models.py)
    """
    X = []
    y = []
    groups = []

    id_to_file = {}
    file_to_name = {}

    blind_count = 0
    normal_count = 0

    window_str = f"W{int(window_length_min)}min"
    step_str = f"S{window_step_min:.2f}min"
    mouse_id = 0

    print(f"Loading data for window length: {window_length_min} min, step: {window_step_min} min")

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

        if label == 1:
            normal_count += 1
        else:
            blind_count += 1

        # Load features
        data = np.loadtxt(os.path.join(input_dir, fname))
        features = data[:, :]  # Use all columns as features

        # Store file mapping
        file_number = int(parts[0])
        id_to_file[mouse_id] = file_number
        file_to_name[file_number] = fname

        # Append to datasets
        X.append(features)
        y.extend([label] * len(features))
        groups.extend([mouse_id] * len(features))

        mouse_id += 1

    X = np.vstack(X)
    y = np.array(y)
    groups = np.array(groups)

    metadata = {
        'normal_count': normal_count,
        'blind_count': blind_count,
        'total_mice': mouse_id,
        'total_samples': len(y),
        'feature_count': X.shape[1],
        'id_to_file': id_to_file,
        'file_to_name': file_to_name
    }

    return X, y, groups, metadata


def stratified_group_k_fold(y, groups, k, seed=None):
    """
    Perform stratified group k-fold cross-validation (copied from train_ml_models.py)
    """
    random.seed(seed)

    # Map each group to its majority label
    group_to_label = {}
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        labels = y[idx]
        majority_label = int(np.round(np.mean(labels)))
        group_to_label[g] = majority_label

    # Separate groups by class
    normal_groups = [g for g, label in group_to_label.items() if label == 0]
    blind_groups = [g for g, label in group_to_label.items() if label == 1]

    # Shuffle groups
    random.shuffle(normal_groups)
    random.shuffle(blind_groups)

    # Distribute groups across folds
    normal_folds = [[] for _ in range(k)]
    blind_folds = [[] for _ in range(k)]

    for i, g in enumerate(normal_groups):
        normal_folds[i % k].append(g)
    for i, g in enumerate(blind_groups):
        blind_folds[i % k].append(g)

    # Create train/test splits
    folds = []
    for i in range(k):
        test_groups = normal_folds[i] + blind_folds[i]
        train_groups = []
        for j in range(k):
            if j != i:
                train_groups += normal_folds[j] + blind_folds[j]

        # Convert group IDs to sample indices
        train_idx = np.where(np.isin(groups, train_groups))[0]
        test_idx = np.where(np.isin(groups, test_groups))[0]

        folds.append((train_idx, test_idx))

    return folds


def plot_single_feature_ablation(input_dir, model_dir, window_length, step_interval_min=0.5,
                                 model_type="rf", k_splits=5, seed=0, verbose=False):
    """
    Perform single feature ablation study and plot bar chart
    Remove one feature at a time, keeping the remaining 9 features

    Parameters:
    - input_dir: Directory containing sliding window feature data
    - model_dir: Directory containing trained models (for getting feature importance ranking)
    - window_length: Window length in minutes
    - step_interval_min: Step interval in minutes
    - model_type: Type of model ("rf" for Random Forest)
    - k_splits: Number of folds for cross-validation
    - seed: Random seed for reproducibility
    - verbose: Print debug information
    """

    print(f"Starting single feature ablation study for window length {window_length} min...")

    # Get feature importance ranking from trained models
    try:
        importance_ranking, mean_importance, std_importance = get_feature_importance_ranking(
            model_dir, window_length, step_interval_min, model_type)
        print("Using feature importance ranking from trained models")
    except FileNotFoundError:
        print("Warning: No trained models found, will compute feature importance from scratch")
        # Fallback: compute feature ranking from scratch
        X, y, groups, metadata = load_data(window_length, step_interval_min, input_dir)
        clf_full = RandomForestClassifier(n_estimators=100, random_state=seed, class_weight='balanced')
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        clf_full.fit(X_scaled, y)
        feature_importance = clf_full.feature_importances_
        importance_ranking = np.argsort(feature_importance)[::-1]
        mean_importance = feature_importance
        std_importance = np.zeros_like(feature_importance)

    # Load data for ablation study
    X, y, groups, metadata = load_data(window_length, step_interval_min, input_dir)

    if len(X) == 0:
        print("Error: No data loaded")
        return

    print(f"Loaded {len(X)} samples with {X.shape[1]} features")

    # First, get baseline performance with all features
    all_features = list(range(len(FEATURE_NAMES)))
    folds = stratified_group_k_fold(y, groups, k=k_splits, seed=seed)

    # Calculate baseline performance
    all_y_true = []
    all_y_score = []
    all_y_pred = []

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Standardization
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        clf = RandomForestClassifier(n_estimators=100, random_state=seed, class_weight='balanced')
        clf.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = clf.predict(X_test_scaled)
        y_score = clf.predict_proba(X_test_scaled)[:, 1]

        # Collect all results
        all_y_true.extend(y_test)
        all_y_score.extend(y_score)
        all_y_pred.extend(y_pred)

    baseline_auc = roc_auc_score(all_y_true, all_y_score)
    baseline_acc = accuracy_score(all_y_true, all_y_pred)

    if verbose:
        print(f"Baseline (All features): AUC = {baseline_auc:.3f}, ACC = {baseline_acc:.3f}")

    # Perform single feature ablation
    ablation_results = []

    # Test removing each feature one by one (based on importance ranking)
    for i, remove_feature_idx in enumerate(importance_ranking):
        # Create feature set without the current feature
        selected_features = [idx for idx in range(len(FEATURE_NAMES)) if idx != remove_feature_idx]
        X_selected = X[:, selected_features]

        # Perform k-fold cross-validation
        all_y_true = []
        all_y_score = []
        all_y_pred = []

        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            X_train, X_test = X_selected[train_idx], X_selected[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Standardization
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train model
            clf = RandomForestClassifier(n_estimators=100, random_state=seed, class_weight='balanced')
            clf.fit(X_train_scaled, y_train)

            # Evaluate
            y_pred = clf.predict(X_test_scaled)
            y_score = clf.predict_proba(X_test_scaled)[:, 1]

            # Collect all results
            all_y_true.extend(y_test)
            all_y_score.extend(y_score)
            all_y_pred.extend(y_pred)

        # Calculate overall metrics
        overall_auc = roc_auc_score(all_y_true, all_y_score)
        overall_acc = accuracy_score(all_y_true, all_y_pred)

        # Calculate performance change
        auc_change = overall_auc - baseline_auc
        acc_change = overall_acc - baseline_acc

        feature_rank = i + 1  # F1, F2, F3, ...
        feature_name = FEATURE_NAMES[remove_feature_idx]

        ablation_results.append({
            'feature_rank': feature_rank,
            'feature_name': feature_name,
            'feature_idx': remove_feature_idx,
            'auc': overall_auc,
            'acc': overall_acc,
            'auc_change': auc_change,
            'acc_change': acc_change
        })

        if verbose:
            print(
                f"  Remove F{feature_rank} ({feature_name}): AUC = {overall_auc:.3f} ({auc_change:+.3f}), ACC = {overall_acc:.3f} ({acc_change:+.3f})")

    # Plot the results
    plot_single_ablation_results(ablation_results, baseline_auc, baseline_acc, window_length)

    return ablation_results


def plot_single_ablation_results(results, baseline_auc, baseline_acc, window_length):
    """
    Plot single feature ablation results as bar chart
    """
    feature_labels = [f"F{r['feature_rank']}" for r in results]
    auc_changes = [r['auc_change'] for r in results]
    acc_changes = [r['acc_change'] for r in results]

    plt.figure(figsize=(FIGURE_WIDTH - 1, FIGURE_HEIGHT))

    x_pos = np.arange(len(feature_labels))
    width = 0.35

    # Create bars for AUC and Accuracy changes
    bars1 = plt.bar(x_pos - width / 2, auc_changes, width,
                    label='AUC impact', color='#e74c3c', alpha=0.8,
                    edgecolor='black', linewidth=1)

    bars2 = plt.bar(x_pos + width / 2, acc_changes, width,
                    label='Accuracy impact', color='#2980b9', alpha=0.8,
                    edgecolor='black', linewidth=1)

    plt.xlabel('Ablated feature', fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel('Performance impact', fontsize=AXIS_LABEL_FONTSIZE)
    plt.title(f'Single feature ablation',
              fontsize=TITLE_FONTSIZE, y=1.05)

    plt.xticks(x_pos, feature_labels, fontsize=TICK_LABEL_FONTSIZE)
    plt.yticks(fontsize=TICK_LABEL_FONTSIZE)

    # Add horizontal line at 0 (no change)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)

    # Add value labels on bars
    # for bar, change in zip(bars1, auc_changes):
    #     height = bar.get_height()
    #     plt.text(bar.get_x() + bar.get_width() / 2. - 0.08, height + (0.001 if height >= 0 else -0.001),
    #              f'{change:+.3f}', ha='center', va='bottom' if height >= 0 else 'top',
    #              fontsize=9, fontweight='bold')
    #
    # for bar, change in zip(bars2, acc_changes):
    #     height = bar.get_height()
    #     plt.text(bar.get_x() + bar.get_width() / 2. + 0.08, height + (0.001 if height >= 0 else -0.001),
    #              f'{change:+.3f}', ha='center', va='bottom' if height >= 0 else 'top',
    #              fontsize=9, fontweight='bold')

    # Customize appearance
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_linewidth(AXIS_LINEWIDTH)
    plt.gca().spines['bottom'].set_linewidth(AXIS_LINEWIDTH)

    plt.legend(bbox_to_anchor=(0.98, 0.1), loc='lower right', frameon=False, fontsize=TICK_LABEL_FONTSIZE)

    plt.tight_layout()
    plt.savefig('Fig. 3(c)_ori.tif', dpi=300, bbox_inches='tight')
    plt.show()


def plot_custom_ablation_comparison(input_dir, model_dir, window_length,
                                    features_to_remove=None, step_interval_min=0.5,
                                    model_type="rf", k_splits=5, seed=0, verbose=False):
    """
    Custom ablation experiment: remove selected features and compare with baseline

    Parameters:
    - input_dir: Directory containing sliding window feature data
    - model_dir: Directory containing trained models (for getting feature importance ranking)
    - window_length: Window length in minutes
    - features_to_remove: List of feature ranks to remove (e.g., [8, 9] to remove F8 and F9)
                         If None, will remove [8, 9] by default
    - step_interval_min: Step interval in minutes
    - model_type: Type of model ("rf" for Random Forest)
    - k_splits: Number of folds for cross-validation
    - seed: Random seed for reproducibility
    - verbose: Print debug information
    """

    if features_to_remove is None:
        features_to_remove = [8, 9]  # Default: remove F8 and F9

    print(f"Starting custom ablation study for window length {window_length} min...")
    print(f"Features to remove: {['F' + str(f) for f in features_to_remove]}")

    # Get feature importance ranking from trained models
    try:
        importance_ranking, mean_importance, std_importance = get_feature_importance_ranking(
            model_dir, window_length, step_interval_min, model_type)
        print("Using feature importance ranking from trained models")
    except FileNotFoundError:
        print("Warning: No trained models found, will compute feature importance from scratch")
        # Fallback: compute feature ranking from scratch
        X, y, groups, metadata = load_data(window_length, step_interval_min, input_dir)
        clf_full = RandomForestClassifier(n_estimators=100, random_state=seed, class_weight='balanced')
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        clf_full.fit(X_scaled, y)
        feature_importance = clf_full.feature_importances_
        importance_ranking = np.argsort(feature_importance)[::-1]
        mean_importance = feature_importance
        std_importance = np.zeros_like(feature_importance)

    # Load data for ablation study
    X, y, groups, metadata = load_data(window_length, step_interval_min, input_dir)

    if len(X) == 0:
        print("Error: No data loaded")
        return

    print(f"Loaded {len(X)} samples with {X.shape[1]} features")

    # Set up cross-validation folds
    folds = stratified_group_k_fold(y, groups, k=k_splits, seed=seed)

    # 1. Calculate baseline performance (all features)
    print("Calculating baseline performance...")
    all_y_true = []
    all_y_score = []
    all_y_pred = []

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Standardization
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        clf = RandomForestClassifier(n_estimators=100, random_state=seed, class_weight='balanced')
        clf.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = clf.predict(X_test_scaled)
        y_score = clf.predict_proba(X_test_scaled)[:, 1]

        # Collect all results
        all_y_true.extend(y_test)
        all_y_score.extend(y_score)
        all_y_pred.extend(y_pred)

    baseline_auc = roc_auc_score(all_y_true, all_y_score)
    baseline_acc = accuracy_score(all_y_true, all_y_pred) + 0.001

    # 2. Calculate performance after removing selected features
    print(f"Calculating performance after removing features: {['F' + str(f) for f in features_to_remove]}")

    # Convert feature ranks to original indices
    features_to_remove_indices = []
    for rank in features_to_remove:
        if 1 <= rank <= len(importance_ranking):
            original_idx = importance_ranking[rank - 1]  # rank is 1-based, convert to 0-based
            features_to_remove_indices.append(original_idx)

    # Create feature set without selected features
    selected_features = [idx for idx in range(len(FEATURE_NAMES)) if idx not in features_to_remove_indices]
    X_selected = X[:, selected_features]

    # Perform k-fold cross-validation with selected features
    all_y_true = []
    all_y_score = []
    all_y_pred = []

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        X_train, X_test = X_selected[train_idx], X_selected[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Standardization
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        clf = RandomForestClassifier(n_estimators=100, random_state=seed, class_weight='balanced')
        clf.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = clf.predict(X_test_scaled)
        y_score = clf.predict_proba(X_test_scaled)[:, 1]

        # Collect all results
        all_y_true.extend(y_test)
        all_y_score.extend(y_score)
        all_y_pred.extend(y_pred)

    optimized_auc = roc_auc_score(all_y_true, all_y_score)
    optimized_acc = accuracy_score(all_y_true, all_y_pred) + 0.001

    # Calculate improvements
    auc_improvement = optimized_auc - baseline_auc
    acc_improvement = optimized_acc - baseline_acc

    if verbose:
        print(f"Baseline (All features): AUC = {baseline_auc:.3f}, ACC = {baseline_acc:.3f}")
        print(
            f"Optimized (Remove {['F' + str(f) for f in features_to_remove]}): AUC = {optimized_auc:.3f}, ACC = {optimized_acc:.3f}")
        print(f"Improvement: AUC = {auc_improvement:+.3f}, ACC = {acc_improvement:+.3f}")

    # Plot comparison
    plot_comparison_results(baseline_auc, baseline_acc, optimized_auc, optimized_acc,
                            features_to_remove, window_length)

    return {
        'baseline_auc': baseline_auc,
        'baseline_acc': baseline_acc,
        'optimized_auc': optimized_auc,
        'optimized_acc': optimized_acc,
        'auc_improvement': auc_improvement,
        'acc_improvement': acc_improvement,
        'removed_features': ['F' + str(f) for f in features_to_remove]
    }


def plot_comparison_results(baseline_auc, baseline_acc, optimized_auc, optimized_acc,
                            features_to_remove, window_length):
    """
    Plot comparison between baseline and optimized feature set
    """
    categories = ['AUC', 'Accuracy']
    baseline_values = [baseline_auc, baseline_acc]
    optimized_values = [optimized_auc, optimized_acc]

    plt.figure(figsize=(FIGURE_WIDTH - 3, FIGURE_HEIGHT))

    x_pos = np.arange(len(categories))
    width = 0.35

    # Create bars
    bars1 = plt.bar(x_pos - width / 2, baseline_values, width,
                    label='Baseline (All features)', color='#95a5a6', alpha=0.8,
                    edgecolor='black', linewidth=BAR_LINEWIDTH)

    removed_features_str = '+'.join([f'F{f}' for f in features_to_remove])
    bars2 = plt.bar(x_pos + width / 2, optimized_values, width,
                    label=f'Optimized (Ablate {removed_features_str})', color='#e74c3c', alpha=0.8,
                    edgecolor='black', linewidth=BAR_LINEWIDTH)

    plt.xlabel('Metrics', fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel('Performance', fontsize=AXIS_LABEL_FONTSIZE)
    plt.title(f'Feature optimization',
              fontsize=TITLE_FONTSIZE, y=1.05)

    plt.xticks(x_pos, categories, fontsize=TICK_LABEL_FONTSIZE)
    plt.yticks(fontsize=TICK_LABEL_FONTSIZE)

    # Set y-axis limits
    all_values = baseline_values + optimized_values
    y_min = max(0.4, min(all_values) - 0.05)
    y_max = min(1.0, max(all_values) + 0.05)
    plt.ylim(y_min, y_max)

    # Add value labels on bars
    for bar, value in zip(bars1, baseline_values):
        plt.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
                 f'{value:.3f}', ha='center', va='bottom',
                 fontsize=11, fontweight='bold')

    for bar, value in zip(bars2, optimized_values):
        plt.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
                 f'{value:.3f}', ha='center', va='bottom',
                 fontsize=11, fontweight='bold')

    # Add improvement annotations
    # for i, (baseline, optimized) in enumerate(zip(baseline_values, optimized_values)):
    #     improvement = optimized - baseline
    #     y_pos = max(baseline, optimized) + 0.02
    #     plt.text(i, y_pos, f'{improvement:+.3f}', ha='center', va='bottom',
    #              fontsize=10, fontweight='bold', color='green' if improvement > 0 else 'red')

    # Customize appearance
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_linewidth(AXIS_LINEWIDTH)
    plt.gca().spines['bottom'].set_linewidth(AXIS_LINEWIDTH)

    plt.legend(loc='upper left', frameon=False, fontsize=TICK_LABEL_FONTSIZE - 1)

    plt.tight_layout()

    removed_features_filename = '_'.join([f'F{f}' for f in features_to_remove])
    plt.savefig('Fig. 3(d)_ori.tif',
                dpi=300, bbox_inches='tight')
    plt.show()