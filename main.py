import os
import easygui
from orchestrator import AudioAnalysisOrchestrator

def main():
    print("="*80)
    print("🎵 AUDIO ANALYSIS SYSTEM ENTRYPOINT")
    print("="*80)

    # Prompt user for the audio directory
    audio_dir = easygui.diropenbox(msg="Select root directory containing audio files", title="Select Input Root")
    if not audio_dir:
        print("Pipeline cancelled: No audio root directory selected.")
        return

    # Select main output folder, defaulting to parent of audio_dir
    base_target = easygui.diropenbox(msg="Select target directory to save output data (logs, features, models, etc.)", title="Select Output Location")
    if not base_target:
        base_target = os.path.dirname(audio_dir)

    # Optional: Find CSV path for XGBoost
    xgb_csv = easygui.fileopenbox(msg="Select tabular feature CSV (Optional, cancel to skip XGBoost)", title="Select XGBoost CSV Data", filetypes=["*.csv"])

    # Prepare directories config
    config = {
        'audio_root_dir': audio_dir,
        'features_target_dir': os.path.join(base_target, 'ExtractedFeatures'),
        'melspec_target_dir': os.path.join(base_target, 'MelSpectrogramImages'),
        'mfcc_target_dir': os.path.join(base_target, 'MFCCImages'),
        'cnn_train_dir': os.path.join(base_target, 'MelSpectrogramImages'), # Default to Mel Specs
        'cnn_model_dir': os.path.join(base_target, 'Models'),
        'cnn_log_dir': os.path.join(base_target, 'Logs'),
        'cnn_checkpoint_dir': os.path.join(base_target, 'Checkpoints'),
        'cnn_batch_size': 10,
        'cnn_epochs': 30, # Default epoch count
        'cnn_model_name': 'audio_classifier_Mel_spec_VCv1.h5',
        'xgb_csv_path': xgb_csv,
        'xgb_plot_path': os.path.join(base_target, 'important_features.png'),
        'knn_plot_path': os.path.join(base_target, 'knn_confusion_matrix.png'),
        'knn_n_neighbors': 10
    }

    orchestrator = AudioAnalysisOrchestrator(config)
    
    # Run the core parts: Feature extractor, Spectrogram image gens, CNN, and optionally XGBoost & KNN
    run_xgb = xgb_csv is not None and os.path.exists(xgb_csv)
    
    # Run the Iris KNN Demo
    run_knn = True

    orchestrator.run_pipeline(
        run_feature_extractor=True,
        run_mel_gen=True,
        run_mfcc_gen=True,
        run_cnn=True,
        run_xgb=run_xgb,
        run_knn=run_knn
    )

if __name__ == "__main__":
    main()
