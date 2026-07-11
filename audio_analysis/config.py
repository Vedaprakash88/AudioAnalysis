import os
import configparser

def load_config(config_path=None):
    """
    Loads configurations from a .ini file and formats them as a pipeline config dictionary.
    """
    if config_path is None:
        # Resolve config.ini relative to this package installation or workspace root
        possible_paths = [
            "config.local.ini",
            os.path.join(os.getcwd(), "config.local.ini"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.local.ini"),
            "config.ini",
            os.path.join(os.getcwd(), "config.ini"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.ini")
        ]
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break
        if not config_path:
            raise FileNotFoundError("Could not locate config.ini in standard paths. Please supply a direct path.")

    print(f"Reading configuration from: {config_path}")
    parser = configparser.ConfigParser()
    parser.read(config_path)

    # Resolve paths
    audio_dir = parser.get('paths', 'audio_root_dir', fallback='')
    out_dir = parser.get('paths', 'output_dir', fallback='')

    # Populate config dictionary matching orchestrator expectations
    config = {
        'audio_root_dir': audio_dir,
        'features_target_dir': os.path.join(out_dir, 'ExtractedFeatures'),
        'melspec_target_dir': os.path.join(out_dir, 'MelSpectrogramImages'),
        'mfcc_target_dir': os.path.join(out_dir, 'MFCCImages'),
        'cnn_train_dir': os.path.join(out_dir, 'MelSpectrogramImages'), # Default, orchestrator will override
        'cnn_model_dir': os.path.join(out_dir, 'Models'),
        'cnn_log_dir': os.path.join(out_dir, 'Logs'),
        'cnn_checkpoint_dir': os.path.join(out_dir, 'Checkpoints'),
        'cnn_batch_size': parser.getint('cnn', 'batch_size', fallback=10),
        'cnn_epochs': parser.getint('cnn', 'epochs', fallback=30),
        'cnn_model_name': parser.get('cnn', 'model_name', fallback='audio_classifier_model.h5'),
        'cnn_feature_type': parser.get('cnn', 'feature_type', fallback='mel'),
        'xgb_csv_path': os.path.join(out_dir, 'ExtractedFeatures', 'extracted_features.csv'), # Default path to compiled CSV
        'xgb_plot_path': os.path.join(out_dir, 'important_features.png'),
        'xgb_params': {
            'max_depth': parser.getint('xgb', 'max_depth', fallback=6),
            'learning_rate': parser.getfloat('xgb', 'learning_rate', fallback=0.1),
            'n_estimators': parser.getint('xgb', 'n_estimators', fallback=500),
            'verbosity': 1,
            'booster': 'gbtree',
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'multi:softmax',
            'eval_metric': 'logloss',
            'nthread': 16
        },
        'knn_plot_path': os.path.join(out_dir, 'knn_confusion_matrix.png'),
        'knn_n_neighbors': parser.getint('knn', 'n_neighbors', fallback=10)
    }

    return config
