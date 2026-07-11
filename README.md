# Audio Analysis and Classification Pipeline - Uni project 2022 (refrshed in 2026)

A clean, modular, and fully Object-Oriented pipeline for audio processing, spectral image generation, and machine learning classification. The system integrates raw feature extraction, visual spectrogram generation, Convolutional Neural Networks (CNNs), and XGBoost/KNN feature classifiers into a unified in-process orchestrator.

## 🚀 Key Features

* **Object-Oriented Design (OOP)**: Replaced procedural scripts with modular, reusable classes for extraction, generation, training, and inference.
* **Installable Package**: Can be installed editably via pip (`pip install -e .`) and imported in any Python script.
* **Auto-compiled Tabular Features**: Automatically aggregates 58 key DSP features (chroma, spectral centroids, HPSS harmony/percussive, tempo, and 20 MFCCs) from raw audio files into a unified GTZAN-style CSV.
* **Unified In-Memory Predictor**: Infers directly from **either raw audio files or pre-rendered images** without creating temporary files on disk.
* **In-Process Orchestration**: Run pipelines programmatically without spawning separate shell subprocesses.
* **Configuration Loading**: Set folders, paths, and hyperparameters inside a `.ini` file.

---

## 🛠️ Installation

1. **Install Package Editably**:
   Activate your virtual environment and run the following command in the repository root directory:
   ```bash
   pip install -e .
   ```
   *(Dependencies include: `librosa`, `numpy`, `matplotlib`, `pandas`, `xgboost`, `scikit-learn`, `easygui`, `tensorflow`, `opencv-python`, `seaborn`, `tqdm`)*

---

## ⚙️ Configuration Setup

To configure directories and training parameters, use the `.ini` configuration format:

1. **`config.ini`**: Provided in the repository with default hyperparameters. Fill in `audio_root_dir` and `output_dir` before executing.
2. **`config.local.ini`** *(Recommended)*: Create a copy named `config.local.ini` at the root. The configuration loader prioritizes `config.local.ini` and it is already ignored in git to keep your local directory paths private.

### Example Configuration:
```ini
[paths]
audio_root_dir = C:/path/to/raw/audio_dataset
output_dir = C:/path/to/save/outputs

[cnn]
feature_type = mel       # Feature representation used: 'mel' or 'mfcc'
batch_size = 10
epochs = 30
model_name = audio_classifier_model.h5

[knn]
n_neighbors = 10

[xgb]
n_estimators = 500
learning_rate = 0.1
max_depth = 6
```

---

## 📖 Usage Manual

### 1. Unified Pipeline (Run "All in One Go")
To process raw audio files, generate visual representations, compile features, and train all classifiers (CNN, XGBoost, KNN) in a single run:

Run the main runner script at the root:
```bash
python main.py
```
*(If directories are not specified in the `.ini` files, easygui graphical prompt boxes will automatically open to guide folder selection).*

You can also run the orchestrator directly in any Python script:
```python
from audio_analysis import load_config, AudioAnalysisOrchestrator

# Load configuration
config = load_config()

# Instantiate and run
orchestrator = AudioAnalysisOrchestrator(config)
orchestrator.run_pipeline(
    run_feature_extractor=True,
    run_mel_gen=True,
    run_mfcc_gen=True,
    run_cnn=True,
    run_xgb=True,
    run_knn=True
)
```

---

### 2. Predictor Inference (Predict individual songs)
You can run inferences on new songs using the `AudioCNNPredictor` class. The predictor dynamically checks file extensions and supports **both raw audio tracks and pre-rendered spectrogram JPEGs**.

```python
from audio_analysis import AudioCNNPredictor

# Initialize predictor with your saved model name and feature type
predictor = AudioCNNPredictor(
    model_path="audio_classifier_model.h5", 
    model_dir="C:/path/to/save/outputs/Models",
    feature_type="mel"
)

# Predict directly from a raw audio file (Mel Spectrogram is rendered in-memory)
class_idx, class_name, probabilities = predictor.get_predicted_class(
    audio_or_image_path="C:/path/to/new_song.wav",
    class_names=["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
)

print(f"Predicted Class: {class_name} (Probabilities: {probabilities})")
```

---

### 3. Running Unit Tests
The repository includes a comprehensive unit test suite to verify configuration parsing, visual generation, classification, and in-memory prediction:

Run all tests via the test runner script:
```bash
python run_tests.py
```
Or directly using Python's unittest module:
```bash
python -m unittest discover -s tests
```

* **`tests/test_config.py`**: Validates configuration `.ini` loading, defaults, and overrides.
* **`tests/test_audio_extractor_generator.py`**: Verifies waveform saving, Mel/MFCC spectrogram rendering, and tabular feature CSV compilation.
* **`tests/test_cnn_classifier.py`**: Checks CNN data loading, model training, evaluation, and dynamic in-memory prediction from raw audio files.
* **`tests/test_xgb_classifier.py`**: Asserts XGBoost classifier training, metric scoring, and cross-validation.
* **`tests/test_knn_classifier.py`**: Tests KNN training, matrix calculations, and evaluation.

---

## 📁 Repository Structure

```text
├── main.py                     # Root entry point script
├── config.ini                  # Template configuration file
├── config.local.ini            # Gitignored local configuration (optional)
├── pyproject.toml              # Library packaging configuration
├── requirements.txt            # System dependencies list
├── tests/                      # Package unit test suite
└── audio_analysis/              # Core library package directory
    ├── __init__.py              # Exposed package imports
    ├── config.py                # Configuration file loader
    ├── orchestrator.py          # Orchestrates training pipelines
    ├── audio_feature_extractor.py  # Extracts waveforms, MFCCs, and compiles features CSV
    ├── audio_image_generator.py    # Converts audio folders to JPEG spectrograms/MFCCs
    ├── audio_cnn_classifier.py     # Prepares datasets and trains the CNN model
    ├── audio_cnn_predictor.py      # Runs inference on raw audio files or JPEG images
    ├── xgb_feature_classifier.py   # Fits, cross-validates and plots XGBoost features
    └── knn_audio_classifier.py     # Prepares and fits the KNN audio features model
```