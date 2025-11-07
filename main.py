
import window_feature_extractor as wfe
import train_ml_models as tmlm
import draw_boxplot as db
import draw_roc as dr
import draw_rf_feature as drff
import opt_feature_group_analysis as optf
import model_comparison as mc
import plot_confusion_matrix as pcm
import dark_histogram as dh


MICE_INFO_DIR = "./Mice information/Mice information.xlsx"
FRAMEWISE_DATA_DIR = "./Framewise position data"
SLIDING_WINDOW_FEATURE_DIR = "./Sliding window features"
ML_MODEL_SAVING_DIR = "./Training models"

MAX_WINDOW_LENGTH = 10

MAX_NORMAL = 25
MAX_BLIND = 25

K_SPLITS = 5


if __name__ == '__main__':
    # ===== Sliding window, extract features and write to file =====
    # for window_length in range(5, 6):  # step_min = 1.0
    # for window_length in range(1, MAX_WINDOW_LENGTH + 1):  # step_min = 0.5
    #     wfe.window_feature_extractor(
    #         mice_info_dir=MICE_INFO_DIR,
    #         input_dir=FRAMEWISE_DATA_DIR,
    #         output_dir=SLIDING_WINDOW_FEATURE_DIR,
    #         window_length_min=float(window_length),
    #         step_min=1.0,
    #         max_normal=MAX_NORMAL,
    #         max_blind=MAX_BLIND
    #     )
    #     print(f"Processing completed for window length: {window_length} minutes\n")
    # print("All window lengths processed successfully!")
    # debug_temp = 0
    # ===== Sliding window, extract features and write to file =====

    # ===== Train ML models =====
    # for window_length in range(5, 6):  # step_min = 1.0
    # for window_length in range(1, MAX_WINDOW_LENGTH + 1):  # step_min = 0.5
    #     tmlm.train_models(
    #         window_length_min=window_length,
    #         window_step_min=1.0,
    #         input_dir=SLIDING_WINDOW_FEATURE_DIR,
    #         k_splits=K_SPLITS,
    #         output_dir=ML_MODEL_SAVING_DIR,
    #         model_type="svm"
    #     )
    # debug_temp = 0
    # ===== Train ML models =====

    # ===== Figure 1 =====
    # use_ml_prediction=False for Fig.1(A)(B), True for Fig.1(C)(D)
    # db.plot_box_by_window_length(
    #     input_dir=SLIDING_WINDOW_FEATURE_DIR,
    #     start_time_min=0,
    #     step_interval_min=0.5,
    #     max_window=MAX_WINDOW_LENGTH,
    #     use_ml_prediction=True,
    #     model_dir=ML_MODEL_SAVING_DIR,
    #     model_type="svm"
    # )
    # debug_temp = 0
    # db.plot_box_by_start_time(
    #     input_dir=SLIDING_WINDOW_FEATURE_DIR,
    #     window_length=5,
    #     max_window_length=15,
    #     step_interval_min=1.0,
    #     use_ml_prediction=True,
    #     model_dir=ML_MODEL_SAVING_DIR,
    #     model_type="svm"
    # )
    # debug_temp = 0
    # ===== Figure 1 =====

    # ===== Figure 2 =====
    # Fig. 2(a)
    # dr.plot_auc_by_window_length(
    #     input_dir=SLIDING_WINDOW_FEATURE_DIR,
    #     ml_model_dir=ML_MODEL_SAVING_DIR,
    #     max_window_length=MAX_WINDOW_LENGTH,
    #     roc_window_length=7,
    #     step_interval_min=0.5,
    #     model_type="svm",
    #     single_feature_index=0,  # 0=dark%
    #     verbose=True
    # )
    # debug_temp = 0

    # Fig. 2(b)
    # pcm.plot_confusion_matrix(
    #     ml_model_dir=ML_MODEL_SAVING_DIR,
    #     model_type="svm",
    #     window_length=7,
    #     step_interval_min=0.5
    # )
    # debug_temp = 0
    # ===== Figure 2 =====

    # ===== Figure 3 =====
    # Fig. 3(a)
    # drff.plot_feature_importance(
    #     model_dir=ML_MODEL_SAVING_DIR,
    #     window_length=5,
    #     step_interval_min=0.5,
    #     model_type="rf"
    # )
    # debug_temp = 0

    # Fig. 3(b)
    # drff.plot_ablation_study(
    #     input_dir=SLIDING_WINDOW_FEATURE_DIR,
    #     model_dir=ML_MODEL_SAVING_DIR,
    #     window_length=5,
    #     step_interval_min=0.5,
    #     model_type="rf",
    #     k_splits=K_SPLITS,
    #     verbose=True
    # )
    # debug_temp = 0

    # Fig. 3(c)
    # drff.plot_single_feature_ablation(
    #     input_dir=SLIDING_WINDOW_FEATURE_DIR,
    #     model_dir=ML_MODEL_SAVING_DIR,
    #     window_length=5,
    #     step_interval_min=0.5,
    #     model_type="rf",
    #     k_splits=K_SPLITS,
    #     verbose=True
    # )

    # Fig. 3(d)
    # drff.plot_custom_ablation_comparison(
    #     input_dir=SLIDING_WINDOW_FEATURE_DIR,
    #     model_dir=ML_MODEL_SAVING_DIR,
    #     window_length=5,
    #     step_interval_min=0.5,
    #     features_to_remove=[8, 9],
    #     verbose=True)
    # debug_temp = 0

    # Fig. 3(e)
    # for window_length in range(1, MAX_WINDOW_LENGTH + 1):
    #     tmlm.train_models(
    #         window_length_min=window_length,
    #         window_step_min=0.5,
    #         input_dir=SLIDING_WINDOW_FEATURE_DIR,
    #         k_splits=K_SPLITS,
    #         output_dir=ML_MODEL_SAVING_DIR,
    #         model_type="rf",
    #         if_remove_features=True,
    #         features_to_remove=[4, 9]  # corresponding to F9, F8
    #     )
    # debug_temp = 0
    # optf.compare_performance(
    #     model_dir=ML_MODEL_SAVING_DIR,
    #     max_window_length=MAX_WINDOW_LENGTH,
    #     step_interval_min=0.5,
    #     model_type="xgb",
    #     verbose=True
    # )
    # ===== Figure 3 =====

    # ===== Figure 4 =====
    # train "rf", "svm", "logistic", "mlp" and "xgb"
    # for window_length in range(1, MAX_WINDOW_LENGTH + 1):
    #     tmlm.train_all_models(
    #             window_length_min=window_length,
    #             window_step_min=0.5,
    #             input_dir=SLIDING_WINDOW_FEATURE_DIR,
    #             k_splits=K_SPLITS,
    #             output_dir=ML_MODEL_SAVING_DIR,
    #             if_remove_features=True,
    #             features_to_remove=[4, 9]
    #         )
    # debug_temp = 0

    # mc.compare_models_performance(
    #     model_dir=ML_MODEL_SAVING_DIR,
    #     max_window_length=10,
    #     if_remove_features=False,  # use opt_features
    #     models=['rf', 'svm', 'xgb', 'logistic', 'mlp']
    # )
    # ===== Figure 4 =====

    # ===== Discussion Figure =====
    dh.calculate_dark_percentage_20min(
        mice_info_dir=MICE_INFO_DIR,
        input_dir=FRAMEWISE_DATA_DIR,
        max_normal=MAX_NORMAL,
        max_blind=MAX_BLIND
    )
    debug_temp = 0

