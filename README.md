# Audio Analysis and Classification Pipeline

A clean, modular, and fully Object-Oriented pipeline for audio processing, spectral image generation, and machine learning classification. The system integrates raw feature extraction, visual spectrogram generation, Convolutional Neural Networks (CNNs), and XGBoost feature classifiers into an unified in-process orchestrator.

## 🚀 Key Features

* **Object-Oriented Design (OOP)**: Replaced procedural scripts with modular, reusable classes for extraction, generation, training, and inference.
* **In-Process Orchestration**: Run pipelines programmatically without spawning separate shell subprocesses, improving efficiency and error handling.
* **Feature Extraction**: Parallelized extraction of raw audio waveforms, MFCCs, Mel Spectrograms, and STFT spectrograms saved directly as NumPy `.npy` files.
* **Visual Representation Generation**: Converts audio datasets into high-fidelity Mel Spectrogram and MFCC JPEG images.
* **CNN Image Classifier**: Trains a custom Convolutional Neural Network on spectral images with TensorBoard logs, checkpoint savings, and GPU memory optimizations.
* **Tabular Feature Classifier**: Fits and evaluates an XGBoost classifier on tabular features with 10-fold cross-validation and feature importance plotting.
* **Interactive Folder Configuration**: Enter directories once using native GUI folders selection prompts.

---

## 📁 Repository Structure

```text
├── main.py                     # Unified entry point for selecting paths and running pipelines
├── orchestrator.py             # Central AudioAnalysisOrchestrator coordinating pipeline steps
├── audio_feature_extractor.py  # AudioFeatureExtractor class (Extracts raw waveform, MFCC, Spectrograms)
├── audio_image_generator.py    # AudioImageGenerator class (Converts audio to JPEG spectrograms)
├── audio_cnn_classifier.py     # AudioCNNClassifier class (Data pipeline, CNN training, and metrics)
├── audio_cnn_predictor.py      # AudioCNNPredictor class (Inference on spectrogram images)
├── xgb_feature_classifier.py   # XGBFeatureClassifier class (XGBoost training, CV, and plot importance)
├── knn_audio_classifier.py     # KNNAudioClassifier class (KNN classification on audio features)
└── requirements.txt            # System dependencies
```

---

## 🛠️ Modular Components Overview

### 1. `AudioFeatureExtractor`
Handles raw digital signal processing (DSP). Loads audio files, extracts audio properties, and saves them as `.npy` arrays.
* **Waveform**: Raw audio signal.
* **MFCC**: Mel-Frequency Cepstral Coefficients.
* **Mel Spectrogram**: Power spectrogram on the Mel scale.
* **Spectrogram**: Short-Time Fourier Transform (STFT) magnitude.

### 2. `AudioImageGenerator`
Consolidates image conversion scripts. Implements multi-threaded conversion of audio tracks to Mel Spectrogram and MFCC images saved as JPEGs using the headless `matplotlib` Agg backend.

### 3. `AudioCNNClassifier`
Prepares TensorFlow image datasets, performs sanitization checks on extensions, normalizes pixel intensities, partitions data (70% Train, 20% Val, 10% Test), builds the sequential CNN model, and exports training metrics history.

### 4. `XGBFeatureClassifier`
Processes tabular audio datasets. Scales features, encodes label targets, fits an `XGBClassifier`, runs 10-fold cross-validation metrics, and plots the top 10 most important features.

---

## 💻 Installation & Usage

### 1. Install Dependencies
Ensure you have Python installed, then install the necessary dependencies:
```bash
pip install -r requirements.txt
```
*(Dependencies include: `librosa`, `numpy`, `matplotlib`, `pandas`, `xgboost`, `scikit-learn`, `easygui`, `tensorflow`, `opencv-python`, `seaborn`, `tqdm`)*

### 2. Execute the Pipeline
Run the main script to start the interactive orchestrator:
```bash
python main.py
```
A series of GUI dialogs will guide you to select your audio dataset folder, output workspace folder, and optional tabular CSV logs.

---

# Problem

Accuracy: 0.92
confusion_matrix:
 [[187   1   8   0   0   4   0   0   3   5]
 [  0 199   0   0   0   3   0   0   1   0]
 [  9   0 165   2   0   5   0   1   2   2]
 [  1   2   2 182   2   1   2   3   2   2]
 [  2   1   5   4 196   1   2   2   3   2]
 [  0   6   6   1   0 179   0   0   0   0]
 [  1   0   1   0   1   0 198   0   0   3]
 [  0   0   0   0   1   0   0 175   2   2]
 [  1   1   8   5   2   1   0   2 190   1]
 [  5   1   8   3   3   3   7   0   2 165]]
classification_report: 
               precision    recall  f1-score   support

           0       0.91      0.90      0.90       208
           1       0.94      0.98      0.96       203
           2       0.81      0.89      0.85       186
           3       0.92      0.91      0.92       199
           4       0.96      0.90      0.93       218
           5       0.91      0.93      0.92       192
           6       0.95      0.97      0.96       204
           7       0.96      0.97      0.96       180
           8       0.93      0.90      0.91       211
           9       0.91      0.84      0.87       197

    accuracy                           0.92      1998
   macro avg       0.92      0.92      0.92      1998
weighted avg       0.92      0.92      0.92      1998

Cross-validated accuracy: 0.66 ± 0.07
Accuracy via score method: 0.92
