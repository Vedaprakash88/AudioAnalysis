# Audio Analysis and Classification Pipeline - Uni project 2022 (refrshed in 2026)

A clean, modular, and fully Object-Oriented pipeline for audio processing, spectral image generation, and machine learning classification. The system integrates raw feature extraction, visual spectrogram generation, Convolutional Neural Networks (CNNs), and XGBoost/KNN feature classifiers into a unified in-process orchestrator.

## 🚀 Key Features

* **Unified Preprocessing (`UnifiedAudioProcessor`)**: Loads and decodes raw audio files exactly once from disk to extract tabular features, Mel JPEGs, and MFCC JPEGs in a single parallel loop. This reduces disk I/O and decoding overhead by up to 66% when running the full pipeline.
* **Object-Oriented Design (OOP)**: Replaced procedural scripts with modular, reusable classes for extraction, generation, training, and inference.
* **Installable Package**: Can be installed editably via pip (`pip install -e .`) and imported in any Python script.
* **Auto-compiled Tabular Features**: Automatically aggregates 58 key DSP features (chroma, spectral centroids, HPSS harmony/percussive, tempo, and 20 MFCCs) from raw audio files into a unified GTZAN-style CSV.
* **Unified In-Memory Predictor**: Infers directly from **either raw audio files or pre-rendered images** without creating temporary files on disk.
* **In-Process Orchestration**: Run pipelines programmatically without spawning separate shell subprocesses.
* **Configuration Loading**: Set folders, paths, and hyperparameters inside a `.ini` file.
* **Auto-Dependency Installation & Robust Encoding**: Setup demo scripts to automatically bootstrap missing dependencies, editable-install the package, and set UTF-8 console output encoding to support decorative logging emojis on Windows.

---

## 📊 Dataset

This pipeline is designed and tested using the **GTZAN Genre Classification** audio dataset, which can be downloaded from [Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification). The dataset consists of 1,000 audio tracks (each 30 seconds long) split evenly across 10 music genres: `blues`, `classical`, `country`, `disco`, `hiphop`, `jazz`, `metal`, `pop`, `reggae`, and `rock`.

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
    ├── unified_processor.py     # Unifies extraction and spectrogram/MFCC image rendering in a single load pass
    ├── audio_feature_extractor.py  # Extracts waveforms, MFCCs, and compiles features CSV (delegates to UnifiedAudioProcessor)
    ├── audio_image_generator.py    # Converts audio folders to JPEG spectrograms/MFCCs (delegates to UnifiedAudioProcessor)
    ├── audio_cnn_classifier.py     # Prepares datasets and trains the CNN model
    ├── audio_cnn_predictor.py      # Runs inference on raw audio files or JPEG images
    ├── xgb_feature_classifier.py   # Fits, cross-validates and plots XGBoost features
    └── knn_audio_classifier.py     # Prepares and fits the KNN audio features model
```

---

## 📋 Terminal Logs

Paste and store run logs/terminal outputs here:

```text
(venv)
[user]@Veda MINGW64 [workspace_root] (main)
$ python demo_full_pipeline.py
================================================================================
🎬 STARTING FULL PATH PIPELINE DEMO & TEST
================================================================================
Reading configuration from: config.local.ini
Overriding CNN epochs to 2 for a quick run.

================================================================================
🚀 AUDIO ANALYSIS PIPELINE ORCHESTRATOR STARTED
================================================================================

--- STEPS 1-3: UNIFIED AUDIO PREPROCESSING ---
Configured options:
  - Tabular features extraction: ENABLED
  - Mel Spectrogram image generation: ENABLED
  - MFCC image generation: ENABLED
Found 999 audio files to process.
Processing dataset in parallel using 16 processes...
Preprocessing audio: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 999/999 [06:25<00:00,  2.59it/s]
Successfully compiled feature table to: [output_dir]\ExtractedFeatures\extracted_features.csv

Preprocessing Summary:
  Successfully processed: 999 files
  Failed: 0 files
  Total: 999
Completed unified audio preprocessing steps.

--- STEP 4: TRAINING CNN CLASSIFIER ---
CNN training configured to use: MEL images from [output_dir]\MelSpectrogramImages
Cleanup completed. Removed 0 invalid files.
Found classes: ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
Found 999 files belonging to 10 classes.
2026-07-18 11:51:15.484360: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2026-07-18 11:51:16.283145: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1616] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 1653 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 3050 Laptop GPU, pci bus id: 0000:01:00.0, compute capability: 8.6
Dataset split completed:
  Total batches: 100
  Train batches: 70
  Val batches: 20
  Test batches: 10
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 256, 256, 16)      448

 max_pooling2d (MaxPooling2D  (None, 128, 128, 16)     0
 )

 conv2d_1 (Conv2D)           (None, 126, 126, 32)      4640

 max_pooling2d_1 (MaxPooling  (None, 63, 63, 32)       0
 2D)

 conv2d_2 (Conv2D)           (None, 61, 61, 16)        4624

 max_pooling2d_2 (MaxPooling  (None, 30, 30, 16)       0
 2D)

 flatten (Flatten)           (None, 14400)             0

 dense (Dense)               (None, 512)               7373312

 dense_1 (Dense)             (None, 128)               65664

 dense_2 (Dense)             (None, 10)                1290

=================================================================
Total params: 7,449,978
Trainable params: 7,449,978
Non-trainable params: 0
_________________________________________________________________
None
Starting training...
Epoch 1/2
2026-07-18 11:51:19.748563: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8101
2026-07-18 11:51:21.583300: I tensorflow/stream_executor/cuda/cuda_blas.cc:1614] TensorFloat-32 will be used for the matrix multiplication. This will only be logged once.
69/70 [============================>.] - ETA: 0s - loss: 2.2727 - accuracy: 0.1362WARNING:absl:Found untraced functions such as _jit_compiled_convolution_op, _jit_compiled_convolution_op, _jit_compiled_convolution_op while saving (showing 3 of 3). These functions will not be directly callable after loading.
70/70 [==============================] - 19s 202ms/step - loss: 2.2733 - accuracy: 0.1343 - val_loss: 2.1578 - val_accuracy: 0.1700
Epoch 2/2
70/70 [==============================] - ETA: 0s - loss: 2.0418 - accuracy: 0.2371WARNING:absl:Found untraced functions such as _jit_compiled_convolution_op, _jit_compiled_convolution_op, _jit_compiled_convolution_op while saving (showing 3 of 3). These functions will not be directly callable after loading.
70/70 [==============================] - 16s 216ms/step - loss: 2.0418 - accuracy: 0.2371 - val_loss: 1.9789 - val_accuracy: 0.2000
Training completed.
Saved performance plots to [output_dir]\Models
Evaluation Accuracy: 0.2020
Model successfully saved to [output_dir]\Models\audio_classifier_Mel_spec_VCv1.h5
Completed CNN classifier training.

--- STEP 5: TRAINING XGBOOST FEATURE CLASSIFIER ---
Loading data from [output_dir]\ExtractedFeatures\extracted_features.csv...
Data preprocessed and split successfully.
Training XGBoost Classifier...
Training completed.

========================================
XGBoost Evaluation Results
========================================
Accuracy: 0.8000
Accuracy via Score: 0.8000
Confusion Matrix:
[[17  0  2  0  0  0  0  0  0  2]
 [ 0 12  0  0  0  0  0  0  0  0]
 [ 0  0 19  0  0  1  0  0  0  4]
 [ 0  0  0 15  2  0  0  1  2  2]
 [ 1  0  0  0 12  0  0  0  1  1]
 [ 0  1  0  0  0 26  0  0  0  0]
 [ 0  0  0  0  0  0 18  0  0  0]
 [ 0  0  0  0  1  0  0 16  2  0]
 [ 2  0  0  1  1  0  1  2 15  0]
 [ 1  0  4  2  1  0  1  0  1 10]]
Classification Report:
              precision    recall  f1-score   support

           0       0.81      0.81      0.81        21
           1       0.92      1.00      0.96        12
           2       0.76      0.79      0.78        24
           3       0.83      0.68      0.75        22
           4       0.71      0.80      0.75        15
           5       0.96      0.96      0.96        27
           6       0.90      1.00      0.95        18
           7       0.84      0.84      0.84        19
           8       0.71      0.68      0.70        22
           9       0.53      0.50      0.51        20

    accuracy                           0.80       200
   macro avg       0.80      0.81      0.80       200
weighted avg       0.80      0.80      0.80       200

Running 10-fold cross-validation...
Cross-validated accuracy: 0.7237 ± 0.0827
Feature importance plot saved to [output_dir]\important_features.png
Completed XGBoost feature classification.

--- STEP 6: TRAINING KNN AUDIO CLASSIFIER ---
Loading data from [output_dir]\ExtractedFeatures\extracted_features.csv...
Data preprocessed and split successfully.
Training KNN Classifier with 10 neighbors...
KNN training completed.

========================================
KNN Audio Classifier Evaluation
========================================
Accuracy: 0.6650
Confusion Matrix:
[[16  0  3  0  0  0  1  0  0  1]
 [ 0 11  0  0  0  1  0  0  0  0]
 [ 1  0 16  2  0  1  0  0  1  3]
 [ 0  0  0 14  0  0  1  2  1  4]
 [ 0  0  0  2 10  0  1  1  1  0]
 [ 1  7  2  0  0 16  0  0  0  1]
 [ 0  0  0  1  0  0 16  0  0  1]
 [ 0  1  3  1  0  0  0 14  0  0]
 [ 1  0  5  1  3  0  1  0 11  0]
 [ 0  0  4  1  1  0  1  2  2  9]]
Classification Report:
              precision    recall  f1-score   support

       blues       0.84      0.76      0.80        21
   classical       0.58      0.92      0.71        12
     country       0.48      0.67      0.56        24
       disco       0.64      0.64      0.64        22
      hiphop       0.71      0.67      0.69        15
        jazz       0.89      0.59      0.71        27
       metal       0.76      0.89      0.82        18
         pop       0.74      0.74      0.74        19
      reggae       0.69      0.50      0.58        22
        rock       0.47      0.45      0.46        20

    accuracy                           0.67       200
   macro avg       0.68      0.68      0.67       200
weighted avg       0.69      0.67      0.67       200

Confusion matrix plot saved to [output_dir]\knn_confusion_matrix.png
Completed KNN audio classification.

================================================================================
🎉 ALL REQUESTED STEPS IN THE PIPELINE COMPLETED SUCCESSFULLY!
================================================================================

--- TESTING ON-THE-FLY INFERENCE ---
Running inference on: [dataset_root]\blues\blues.00000.wav
Loading model from: [output_dir]\Models\audio_classifier_Mel_spec_VCv1.h5
Model loaded successfully.

========================================
🔮 PREDICTION RESULT
========================================
Song Path: [dataset_root]\blues\blues.00000.wav
Predicted Class Index: 4
Predicted Class Name:  hiphop
Prediction Probabilities: [0.08797272 0.05863929 0.1034588  0.12052273 0.13836163 0.12594415
 0.04840442 0.09942099 0.1040507  0.11322454]
================================================================================


```