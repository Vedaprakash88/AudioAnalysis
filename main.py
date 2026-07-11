import os
import easygui
from audio_analysis import load_config, AudioAnalysisOrchestrator

def main():
    print("="*80)
    print("🎵 AUDIO ANALYSIS SYSTEM ENTRYPOINT")
    print("="*80)

    # 1. Load config from config.ini
    try:
        config = load_config()
    except Exception as e:
        print(f"Warning loading config.ini: {e}. Fallback to manual GUI selections.")
        config = {}

    # 2. Check if folders are configured/exist, otherwise prompt user graphically
    audio_dir = config.get('audio_root_dir', '')
    if not audio_dir or not os.path.exists(audio_dir):
        audio_dir = easygui.diropenbox(msg="Select root directory containing audio files", title="Select Input Root")
        if not audio_dir:
            print("Pipeline cancelled: No audio root directory selected.")
            return
        config['audio_root_dir'] = audio_dir

    # Set features target dir
    features_target_dir = config.get('features_target_dir', '')
    if not features_target_dir:
        base_target = easygui.diropenbox(msg="Select target directory to save output data (logs, features, models, etc.)", title="Select Output Location")
        if not base_target:
            base_target = os.path.dirname(audio_dir)
        
        # Populate folders
        config['features_target_dir'] = os.path.join(base_target, 'ExtractedFeatures')
        config['melspec_target_dir'] = os.path.join(base_target, 'MelSpectrogramImages')
        config['mfcc_target_dir'] = os.path.join(base_target, 'MFCCImages')
        config['cnn_train_dir'] = os.path.join(base_target, 'MelSpectrogramImages')
        config['cnn_model_dir'] = os.path.join(base_target, 'Models')
        config['cnn_log_dir'] = os.path.join(base_target, 'Logs')
        config['cnn_checkpoint_dir'] = os.path.join(base_target, 'Checkpoints')
        config['xgb_csv_path'] = os.path.join(base_target, 'ExtractedFeatures', 'extracted_features.csv')
        config['xgb_plot_path'] = os.path.join(base_target, 'important_features.png')
        config['knn_plot_path'] = os.path.join(base_target, 'knn_confusion_matrix.png')

    # Update xgb parameters csv path just in case
    if 'xgb_csv_path' not in config:
        config['xgb_csv_path'] = os.path.join(os.path.dirname(config['features_target_dir']), 'ExtractedFeatures', 'extracted_features.csv')

    print(f"Configured Audio Root: {config['audio_root_dir']}")
    print(f"Configured Output Target: {os.path.dirname(config['features_target_dir'])}")
    print(f"CNN Feature Type: {config.get('cnn_feature_type', 'mel').upper()}")

    # 3. Instantiate and run orchestrator
    orchestrator = AudioAnalysisOrchestrator(config)
    orchestrator.run_pipeline(
        run_feature_extractor=True,
        run_mel_gen=True,
        run_mfcc_gen=True,
        run_cnn=True,
        run_xgb=True,
        run_knn=True
    )

if __name__ == "__main__":
    main()
