import os
from .audio_feature_extractor import AudioFeatureExtractor
from .audio_image_generator import AudioImageGenerator
from .audio_cnn_classifier import AudioCNNClassifier
from .xgb_feature_classifier import XGBFeatureClassifier
from .knn_audio_classifier import KNNAudioClassifier

class AudioAnalysisOrchestrator:
    """
    Orchestrates the entire audio processing, feature extraction, image generation,
    and classifier training pipelines.
    """
    def __init__(self, config):
        """
        Initializes the orchestrator with a configuration dictionary of directory paths and parameters.
        """
        self.config = config

    def run_pipeline(self, 
                      run_feature_extractor=True, 
                      run_mel_gen=True, 
                      run_mfcc_gen=True, 
                      run_cnn=True, 
                      run_xgb=True, 
                      run_knn=True):
        """
        Runs the specified segments of the machine learning pipeline in-process.
        """
        print("\n" + "="*80)
        print("🚀 AUDIO ANALYSIS PIPELINE ORCHESTRATOR STARTED")
        print("="*80 + "\n")

        # Step 1: Raw Feature Extraction & Tabular CSV compilation (.npy and .csv files)
        if run_feature_extractor:
            print("--- STEP 1: EXTRACTING AUDIO FEATURES (.npy & .csv) ---")
            audio_root = self.config.get('audio_root_dir')
            features_target = self.config.get('features_target_dir')
            
            if not audio_root or not features_target:
                print("❌ Skipped Step 1: Missing 'audio_root_dir' or 'features_target_dir' in config.")
            else:
                extractor = AudioFeatureExtractor(root_dir=audio_root, target_dir=features_target)
                extractor.extract_all()
                print("Completed raw feature extraction and tabular compilation.\n")

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
            feat_type = self.config.get('cnn_feature_type', 'mel').lower()
            
            # Map input directories dynamically based on feature selection
            if feat_type == 'mfcc':
                cnn_train_dir = self.config.get('mfcc_target_dir')
            else:
                cnn_train_dir = self.config.get('melspec_target_dir')
                
            cnn_model_dir = self.config.get('cnn_model_dir')
            cnn_log_dir = self.config.get('cnn_log_dir')
            cnn_chk_pt_dir = self.config.get('cnn_checkpoint_dir')

            if not all([cnn_train_dir, cnn_model_dir, cnn_log_dir, cnn_chk_pt_dir]):
                print("❌ Skipped Step 4: Missing CNN directory paths in config.")
            elif not os.path.exists(cnn_train_dir) or not os.listdir(cnn_train_dir):
                print(f"❌ Skipped Step 4: Training directory '{cnn_train_dir}' is empty or does not exist.")
            else:
                print(f"CNN training configured to use: {feat_type.upper()} images from {cnn_train_dir}")
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
            
            if not xgb_csv or not os.path.exists(xgb_csv):
                print(f"❌ Skipped Step 5: Tabular feature CSV not found at '{xgb_csv}'. Run Step 1 first.")
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
            
            if not xgb_csv or not os.path.exists(xgb_csv):
                print(f"❌ Skipped Step 6: Tabular feature CSV not found at '{xgb_csv}'. Run Step 1 first.")
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
