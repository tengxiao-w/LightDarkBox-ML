import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
import random
import joblib
import json

# Try to import XGBoost, handle if not installed
try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not installed. XGB model will not be available.")


def get_model(model_type, seed=0):
    """
    Get machine learning model based on model type

    Parameters:
    - model_type: Type of model ("rf", "svm", "xgb", "logistic", "mlp")
    - seed: Random seed for reproducibility

    Returns:
    - Initialized model object
    """
    model_type = model_type.lower()

    if model_type == "rf":
        return RandomForestClassifier(
            n_estimators=100,
            random_state=seed,
            class_weight='balanced'
        )

    elif model_type == "svm":
        return SVC(
            kernel='rbf',
            probability=True,  # Enable probability estimates
            random_state=seed,
            class_weight='balanced',
            gamma='scale'
        )

    elif model_type == "xgb":
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed. Please install it using: pip install xgboost")
        return xgb.XGBClassifier(
            n_estimators=100,
            random_state=seed,
            eval_metric='logloss'
        )

    elif model_type == "logistic":
        return LogisticRegression(
            random_state=seed,
            class_weight='balanced',
            max_iter=1000
        )

    elif model_type == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(100, 50),
            random_state=seed,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10
        )

    else:
        available_models = ["rf", "svm", "xgb", "logistic", "mlp"]
        if not XGBOOST_AVAILABLE:
            available_models.remove("xgb")
        raise ValueError(f"Unsupported model type: {model_type}. Currently supported: {available_models}")


def train_models(window_length_min, window_step_min, input_dir, k_splits=5, output_dir="trained_models",
                 model_type="rf", seed=0, if_remove_features=False, features_to_remove=None):
    """
    Train machine learning models using k-fold cross-validation and save them

    Parameters:
    - window_length_min: Window length in minutes
    - window_step_min: Window step interval in minutes
    - input_dir: Directory containing sliding window data
    - k_splits: Number of folds for cross-validation
    - output_dir: Directory to save trained models
    - model_type: Type of model to train ("rf", "svm", "xgb", "logistic", "mlp")
    - seed: Random seed for reproducibility
    - if_remove_features: Whether to remove specified features
    - features_to_remove: List of feature indices to remove (0-based indexing)

    Returns:
    - Dictionary containing training results and metrics
    """
    # Validate model type
    available_models = ["rf", "svm", "xgb", "logistic", "mlp"]
    if not XGBOOST_AVAILABLE and model_type.lower() == "xgb":
        raise ImportError("XGBoost is not installed. Please install it using: pip install xgboost")

    # Set default value for features_to_remove
    if features_to_remove is None:
        features_to_remove = []

    # Create output directory with model-specific subfolder
    if if_remove_features and features_to_remove:
        model_output_dir = os.path.join(output_dir, f"{model_type}(opt_features)")
        print(f"Feature removal enabled. Removing features at indices: {features_to_remove}")
    else:
        model_output_dir = os.path.join(output_dir, model_type)
        if_remove_features = False  # Override to False if no features to remove

    os.makedirs(model_output_dir, exist_ok=True)

    # Load data
    X, y, groups, metadata = load_data(window_length_min, window_step_min, input_dir)

    print(f"Data loaded: {len(X)} samples, {X.shape[1]} features")

    # Remove specified features if requested
    if if_remove_features and features_to_remove:
        # Validate feature indices
        valid_indices = [i for i in features_to_remove if 0 <= i < X.shape[1]]
        invalid_indices = [i for i in features_to_remove if i not in valid_indices]

        if invalid_indices:
            print(f"Warning: Invalid feature indices {invalid_indices} will be ignored.")

        if valid_indices:
            # Remove features
            remaining_features = [i for i in range(X.shape[1]) if i not in valid_indices]
            X = X[:, remaining_features]

            print(f"Removed {len(valid_indices)} features: {valid_indices}")
            print(f"Remaining features: {len(remaining_features)} ({remaining_features})")

            # Update metadata to record feature removal
            metadata['feature_removal'] = {
                'removed_indices': valid_indices,
                'remaining_indices': remaining_features,
                'original_feature_count': metadata['feature_count'],
                'new_feature_count': X.shape[1]
            }
        else:
            print("No valid features to remove. Proceeding with all features.")
            if_remove_features = False

    print(f"Normal mice: {metadata['normal_count']}, Blind mice: {metadata['blind_count']}")

    # Perform stratified group k-fold cross-validation
    folds = stratified_group_k_fold(y, groups, k=k_splits, seed=seed)

    # Training results storage
    training_results = {
        'window_length_min': window_length_min,
        'window_step_min': window_step_min,
        'k_splits': k_splits,
        'model_type': model_type,
        'feature_removal_enabled': if_remove_features,
        'removed_features': features_to_remove if if_remove_features else [],
        'metadata': metadata,
        'fold_results': []
    }

    # Storage for all predictions
    all_predictions = []

    print(f"\nStarting {k_splits}-fold cross-validation training with {model_type.upper()}...")

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        # print(f"\nTraining Fold {fold_idx + 1}/{k_splits}...")

        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Get file information for this fold
        train_mouse_ids = np.unique(groups[train_idx])
        test_mouse_ids = np.unique(groups[test_idx])
        train_files = [metadata['id_to_file'][mid] for mid in train_mouse_ids]
        test_files = [metadata['id_to_file'][mid] for mid in test_mouse_ids]

        # Standardization
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        clf = get_model(model_type, seed)

        # Special handling for XGBoost to handle class imbalance
        if model_type.lower() == "xgb":
            # Calculate scale_pos_weight for class imbalance
            n_negative = np.sum(y_train == 0)
            n_positive = np.sum(y_train == 1)
            scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1
            clf.set_params(scale_pos_weight=scale_pos_weight)

        clf.fit(X_train_scaled, y_train)

        # Evaluate on test set
        y_pred = clf.predict(X_test_scaled)
        y_score = clf.predict_proba(X_test_scaled)[:, 1]

        # Save sample-level predictions
        test_groups_unique = np.unique(groups[test_idx])
        for sample_idx, (orig_idx, pred_prob, true_label) in enumerate(zip(test_idx, y_score, y_test)):
            mouse_id = groups[orig_idx]
            file_number = metadata['id_to_file'][mouse_id]
            file_name = metadata['file_to_name'][file_number]

            # Calculate sample index within the file
            mouse_samples = test_idx[groups[test_idx] == mouse_id]
            sample_index = np.where(mouse_samples == orig_idx)[0][0]

            # Determine group from filename
            group = 'normal' if '_normal_' in file_name.lower() else 'blind'

            all_predictions.append({
                'file_name': file_name,
                'mouse_id': file_number,
                'group': group,
                'sample_index': sample_index,
                'true_label': true_label,
                'predicted_prob': pred_prob,
                'fold': fold_idx
            })

        # Calculate metrics
        auc_score = roc_auc_score(y_test, y_score)
        accuracy = np.mean(y_pred == y_test)

        # Save model and scaler in model-specific directory
        model_path = os.path.join(model_output_dir,
                                  f"model_{model_type}_w{window_length_min}min_s{window_step_min}min_fold{fold_idx}.joblib")
        scaler_path = os.path.join(model_output_dir,
                                   f"scaler_{model_type}_w{window_length_min}min_s{window_step_min}min_fold{fold_idx}.joblib")
        joblib.dump(clf, model_path)
        joblib.dump(scaler, scaler_path)

        # Store fold results
        fold_result = {
            'fold_idx': fold_idx,
            'train_files': sorted(train_files),
            'test_files': sorted(test_files),
            'auc_score': auc_score,
            'accuracy': accuracy,
            'model_path': model_path,
            'scaler_path': scaler_path
        }
        training_results['fold_results'].append(fold_result)

        print(f"  AUC: {auc_score:.3f}, Accuracy: {accuracy:.3f}")

    # Calculate overall metrics using all test predictions combined
    predictions_df = pd.DataFrame(all_predictions)
    overall_auc = roc_auc_score(predictions_df['true_label'], predictions_df['predicted_prob'])
    overall_predictions = (predictions_df['predicted_prob'] >= 0.5).astype(int)
    overall_accuracy = np.mean(overall_predictions == predictions_df['true_label'])

    # Also keep fold-wise metrics for reference
    all_auc = [result['auc_score'] for result in training_results['fold_results']]
    all_acc = [result['accuracy'] for result in training_results['fold_results']]

    training_results['summary'] = {
        'overall_auc': overall_auc,
        'overall_accuracy': overall_accuracy,
        'mean_auc': np.mean(all_auc),
        'std_auc': np.std(all_auc),
        'mean_accuracy': np.mean(all_acc),
        'std_accuracy': np.std(all_acc)
    }

    # Save predictions as CSV in model-specific directory
    predictions_df = pd.DataFrame(all_predictions)
    predictions_path = os.path.join(model_output_dir,
                                    f"predictions_{model_type}_w{window_length_min}min_s{window_step_min}min.csv")
    predictions_df.to_csv(predictions_path, index=False)

    # Save training results in model-specific directory
    results_path = os.path.join(model_output_dir,
                                f"training_results_{model_type}_w{window_length_min}min_s{window_step_min}min.json")
    with open(results_path, 'w') as f:
        json.dump(training_results, f, indent=2)

    print(f"\n=== Training Summary ===")
    print(f"Model Type: {model_type}")
    if if_remove_features:
        print(f"Feature Removal: Enabled (removed indices: {features_to_remove})")
    print(f"Window Length: {window_length_min} min, Step: {window_step_min} min")
    print(f"Overall AUC: {training_results['summary']['overall_auc']:.3f}")
    print(f"Overall Accuracy: {training_results['summary']['overall_accuracy']:.3f}")
    print(f"Models and results saved to: {model_output_dir}")
    print(f"Predictions saved to: {predictions_path}")

    return training_results


def load_data(window_length_min, window_step_min, input_dir):
    """
    Load and prepare data for training

    Parameters:
    - window_length_min: Window length in minutes
    - window_step_min: Window step interval in minutes
    - input_dir: Directory containing sliding window data

    Returns:
    - X: Feature matrix
    - y: Labels
    - groups: Group assignments for each sample
    - metadata: Additional information about the dataset
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
    Perform stratified group k-fold cross-validation
    Ensures that samples from the same mouse are in the same fold
    and maintains class balance across folds
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


# Example usage for batch training all models
def train_all_models(window_length_min, window_step_min, input_dir, k_splits=5, output_dir="trained_models",
                     seed=0, if_remove_features=False, features_to_remove=None):
    """
    Train all available models with the same parameters

    Parameters:
    - window_length_min: Window length in minutes
    - window_step_min: Window step interval in minutes
    - input_dir: Directory containing sliding window data
    - k_splits: Number of folds for cross-validation
    - output_dir: Directory to save trained models
    - seed: Random seed for reproducibility
    - if_remove_features: Whether to remove specified features
    - features_to_remove: List of feature indices to remove (0-based indexing)

    Returns:
    - Dictionary containing results for all models
    """
    models = ["rf", "svm", "logistic", "mlp"]
    if XGBOOST_AVAILABLE:
        models.append("xgb")

    all_results = {}

    print(f"Training all available models: {models}")
    if if_remove_features and features_to_remove:
        print(f"Feature removal enabled for all models. Removing features at indices: {features_to_remove}")
    print("=" * 60)

    for model_type in models:
        print(f"\n{'=' * 20} Training {model_type.upper()} {'=' * 20}")
        try:
            result = train_models(
                window_length_min=window_length_min,
                window_step_min=window_step_min,
                input_dir=input_dir,
                k_splits=k_splits,
                output_dir=output_dir,
                model_type=model_type,
                seed=seed,
                if_remove_features=if_remove_features,
                features_to_remove=features_to_remove
            )
            all_results[model_type] = result
        except Exception as e:
            print(f"Error training {model_type}: {e}")
            all_results[model_type] = {'error': str(e)}

    # Print comparison summary
    print(f"\n{'=' * 20} MODEL COMPARISON SUMMARY {'=' * 20}")
    print(f"{'Model':<10} {'Overall AUC':<12} {'Overall Acc':<12} {'Mean AUC':<10} {'Std AUC':<8}")
    print("-" * 60)

    for model_type in models:
        if model_type in all_results and 'error' not in all_results[model_type]:
            summary = all_results[model_type]['summary']
            print(f"{model_type.upper():<10} {summary['overall_auc']:<12.3f} {summary['overall_accuracy']:<12.3f} "
                  f"{summary['mean_auc']:<10.3f} {summary['std_auc']:<8.3f}")
        else:
            print(f"{model_type.upper():<10} {'ERROR':<12} {'ERROR':<12} {'ERROR':<10} {'ERROR':<8}")

    return all_results