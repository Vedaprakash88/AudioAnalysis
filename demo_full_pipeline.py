import os
from audio_analysis import load_config, AudioAnalysisOrchestrator, AudioCNNPredictor

def run_test():
    print("=" * 80)
    print("🎬 STARTING FULL PATH PIPELINE DEMO & TEST")
    print("=" * 80)

    # 1. Load configuration (prioritizing config.local.ini)
    try:
        config = load_config()
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return

    # Check if directories exist
    audio_dir = config['audio_root_dir']
    if not os.path.exists(audio_dir):
        print(f"Error: Audio directory does not exist at {audio_dir}")
        print("Please configure paths in config.local.ini first.")
        return

    # For testing purposes, let's temporarily set epochs to 2 in the config copy
    # to make the test run quickly (feel free to remove this to run a full 30 epochs!)
    config_copy = config.copy()
    config_copy['cnn_epochs'] = 2
    print(f"Overriding CNN epochs to {config_copy['cnn_epochs']} for a quick run.")

    # 2. Run the unified pipeline orchestrator
    orchestrator = AudioAnalysisOrchestrator(config_copy)
    orchestrator.run_pipeline(
        run_feature_extractor=True,  # Extracted to extracted_features.csv
        run_mel_gen=True,            # Generates Mel Spec JPEGs
        run_mfcc_gen=True,           # Generates MFCC JPEGs
        run_cnn=True,                # Trains CNN on spectrographs
        run_xgb=True,                # Trains XGBoost on compiled CSV
        run_knn=True                 # Trains KNN on compiled CSV
    )

    # 3. Test on-the-fly Predictor Inference on raw audio file
    print("--- TESTING ON-THE-FLY INFERENCE ---")
    model_name = config_copy['cnn_model_name']
    model_dir = config_copy['cnn_model_dir']
    feature_type = config_copy.get('cnn_feature_type', 'mel')
    
    model_path = os.path.join(model_dir, model_name)
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    # Search for an audio file to run inference on
    found_audio = None
    for root, _, files in os.walk(audio_dir):
        for f in files:
            if f.lower().endswith(('.wav', '.mp3')):
                found_audio = os.path.join(root, f)
                break
        if found_audio:
            break

    if not found_audio:
        print("No audio track found in dataset to test inference.")
        return

    print(f"Running inference on: {found_audio}")
    predictor = AudioCNNPredictor(
        model_path=model_path,
        feature_type=feature_type
    )

    # Automatically infer class names from visual directories
    class_names = sorted(os.listdir(config_copy['melspec_target_dir']))
    class_idx, class_name, probs = predictor.get_predicted_class(
        audio_or_image_path=found_audio,
        class_names=class_names
    )

    print("\n" + "=" * 40)
    print("🔮 PREDICTION RESULT")
    print("=" * 40)
    print(f"Song Path: {found_audio}")
    print(f"Predicted Class Index: {class_idx}")
    print(f"Predicted Class Name:  {class_name}")
    print(f"Prediction Probabilities: {probs}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    run_test()
