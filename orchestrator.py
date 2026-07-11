import os
from audio_feature_extractor import AudioFeatureExtractor
from audio_image_generator import AudioImageGenerator
from audio_cnn_classifier import AudioCNNClassifier
from xgb_feature_classifier import XGBFeatureClassifier
from knn_audio_classifier import KNNAudioClassifier

class AudioAnalysisOrchestrator:
    """
    Orchestrates the entire audio processing, feature extraction, image generation,
    and classifier training pipelines.
    """
    def __init__(self, config):
        """
        Initializes the orchestrator with a configuration dictionary of directory paths.
        
        Required/Optional keys in config:
        - audio_root_dir: Folder containing subclass folders with audio files
        - features_target_dir: Folder where raw numpy (.npy) features will be saved
        - melspec_target_dir: Folder where Mel Spectrogram images will be saved
        - mfcc_target_dir: Folder where MFCC images will be saved
        - cnn_train_dir: Folder of images used for CNN training (e.g., melspec or mfcc target dir)
        - cnn_model_dir: Folder to output the final trained CNN model
        - cnn_log_dir: Folder for Tensorboard logs
        - cnn_checkpoint_dir: Folder for training checkpoints
        - cnn_batch_size: Batch size for CNN training (default: 10)
        - cnn_epochs: Number of epochs for CNN training (default: 50)
        - cnn_model_name: Filename of the saved CNN model (default: 'audio_classifier_model.h5')
        - xgb_csv_path: Path to the feature CSV file (for XGBoost)
        - xgb_plot_path: Path to save feature importance plot
        - xgb_params: Dictionary of parameters for XGBClassifier
        - knn_n_neighbors: Number of neighbors for KNN Iris classifier (default: 10)
        - knn_plot_path: Path to save KNN confusion matrix plot
        """
        self.config = config

    def run_pipeline(self, 
                     run_feature_extractor=True, 
                     run_mel_gen=True, 
                     run_mfcc_gen=True, 
                     run_cnn=True, 
                     run_xgb=False, 
                     run_knn=False):
        """
        Runs the specified segments of the machine learning pipeline in-process.
        """
        print("\n" + "="*80)
        print("🚀 AUDIO ANALYSIS PIPELINE ORCHESTRATOR STARTED")
        print("="*80 + "\n")

        # Step 1: Raw Feature Extraction (.npy files)
        if run_feature_extractor:
            print("--- STEP 1: EXTRACTING AUDIO FEATURES (.npy) ---")
            audio_root = self.config.get('audio_root_dir')
            features_target = self.config.get('features_target_dir')
            
            if not audio_root or not features_target:
                print("❌ Skipped Step 1: Missing 'audio_root_dir' or 'features_target_dir' in config.")
            else:
                extractor = AudioFeatureExtractor(root_dir=audio_root, target_dir=features_target)
                extractor.extract_all()
                print("Completed raw feature extraction.\n")

        # Step 2: Generate Mel Spectrogram Images
        if run_mel_gen:
            print("--- STEP 2: GENERATING MEL SPECTROGRAM IMAGES ---")
            audio_root = self.config.get('audio_root_dir')
            melspec_target = self.config.get('melspec_target_dir')

            if not audio_root or not melspec_target:
                print("❌ Skipped Step 2: Missing 'audio_root_dir' or 'melspec_target_dir' in config.")
            else:
                generator = AudioImageGenerator(root_dir=audio_root, target_dir=melspec_target)
                generator.generate_mel_spectrograms()
                print("Completed Mel Spectrogram image generation.\n")

        # Step 3: Generate MFCC Images
        if run_mfcc_gen:
            print("--- STEP 3: GENERATING MFCC IMAGES ---")
            audio_root = self.config.get('audio_root_dir')
            mfcc_target = self.config.get('mfcc_target_dir')

            if not audio_root or not mfcc_target:
                print("❌ Skipped Step 3: Missing 'audio_root_dir' or 'mfcc_target_dir' in config.")
            else:
                generator = AudioImageGenerator(root_dir=audio_root, target_dir=mfcc_target)
                generator.generate_mfccs()
                print("Completed MFCC image generation.\n")

        # Step 4: Train CNN Image Classifier
        if run_cnn:
            print("--- STEP 4: TRAINING CNN CLASSIFIER ---")
            cnn_train_dir = self.config.get('cnn_train_dir')
            cnn_model_dir = self.config.get('cnn_model_dir')
            cnn_log_dir = self.config.get('cnn_log_dir')
            cnn_chk_pt_dir = self.config.get('cnn_checkpoint_dir')

            if not all([cnn_train_dir, cnn_model_dir, cnn_log_dir, cnn_chk_pt_dir]):
                print("❌ Skipped Step 4: Missing CNN directory paths in config.")
            else:
                classifier = AudioCNNClassifier(
                    data_dir=cnn_train_dir,
                    model_dir=cnn_model_dir,
                    log_dir=cnn_log_dir,
                    checkpoint_dir=cnn_chk_pt_dir,
                    batch_size=self.config.get('cnn_batch_size', 10),
                    epochs=self.config.get('cnn_epochs', 50)
                )
                classifier.prepare_data()
                classifier.train(save_plots=True)
                classifier.evaluate()
                
                model_name = self.config.get('cnn_model_name', 'audio_classifier_model.h5')
                classifier.save_model(model_name)
                print("Completed CNN classifier training.\n")

        # Step 5: Train XGBoost Classifier
        if run_xgb:
            print("--- STEP 5: TRAINING XGBOOST FEATURE CLASSIFIER ---")
            xgb_csv = self.config.get('xgb_csv_path')
            
            if not xgb_csv:
                print("❌ Skipped Step 5: Missing 'xgb_csv_path' in config.")
            else:
                xgb_clf = XGBFeatureClassifier(
                    csv_path=xgb_csv,
                    model_params=self.config.get('xgb_params')
                )
                xgb_clf.train(verbose=True)
                xgb_clf.evaluate(run_cv=True)
                
                xgb_plot = self.config.get('xgb_plot_path', 'important_features.png')
                xgb_clf.plot_feature_importance(output_path=xgb_plot)
                print("Completed XGBoost feature classification.\n")

        # Step 6: Train KNN Audio Classifier
        if run_knn:
            print("--- STEP 6: TRAINING KNN AUDIO CLASSIFIER ---")
            xgb_csv = self.config.get('xgb_csv_path')
            if not xgb_csv:
                print("❌ Skipped Step 6: Missing 'xgb_csv_path' in config.")
            else:
                knn_plot = self.config.get('knn_plot_path', 'knn_confusion_matrix.png')
                knn_clf = KNNAudioClassifier(
                    csv_path=xgb_csv,
                    n_neighbors=self.config.get('knn_n_neighbors', 10)
                )
                knn_clf.train()
                knn_clf.evaluate(save_plot_path=knn_plot)
                print("Completed KNN audio classification.\n")

        print("="*80)
        print("🎉 ALL REQUESTED STEPS IN THE PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80 + "\n")


if __name__ == "__main__":
    import easygui

    print("🚀 NATIVE PIPELINE TEST RUNNER")
    
    audio_dir = easygui.diropenbox(msg="Select root folder containing audio subfolders", title="Select Input Audio")
    if not audio_dir:
        print("Cancelled.")
        exit(0)

    # Ask for target directory to output logs/features/models under
    base_target = easygui.diropenbox(msg="Select main output target directory", title="Select Output Location")
    if not base_target:
        base_target = os.path.dirname(audio_dir)

    # Establish default subdirectories
    config = {
        'audio_root_dir': audio_dir,
        'features_target_dir': os.path.join(base_target, 'ExtractedFeatures'),
        'melspec_target_dir': os.path.join(base_target, 'MelSpectrogramImages'),
        'mfcc_target_dir': os.path.join(base_target, 'MFCCImages'),
        'cnn_train_dir': os.path.join(base_target, 'MelSpectrogramImages'), # Train on Mel Specs by default
        'cnn_model_dir': os.path.join(base_target, 'Models'),
        'cnn_log_dir': os.path.join(base_target, 'Logs'),
        'cnn_checkpoint_dir': os.path.join(base_target, 'Checkpoints'),
        'cnn_batch_size': 10,
        'cnn_epochs': 5, # Low epoch for default test
        'cnn_model_name': 'audio_classifier_Mel_spec_VCv1.h5',
    }

    orchestrator = AudioAnalysisOrchestrator(config)
    orchestrator.run_pipeline(
        run_feature_extractor=True,
        run_mel_gen=True,
        run_mfcc_gen=True,
        run_cnn=True,
        run_xgb=False, # XGB needs tabular CSV
        run_knn=False
    )
