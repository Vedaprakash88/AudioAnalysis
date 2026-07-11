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
├── knn_iris_classifier.py      # KNNIrisClassifier class (KNN baseline classification)
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
