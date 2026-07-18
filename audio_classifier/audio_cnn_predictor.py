import os
import cv2
import numpy as np
import tensorflow as tf

try:
    from tensorflow.keras.models import load_model
except ImportError:
    from tensorflow.python.keras.models import load_model

class AudioCNNPredictor:
    """
    Inference helper to predict class probabilities from EITHER raw audio files (WAV, MP3, etc.)
    OR pre-rendered spectrogram images (JPEG, PNG).
    """
    def __init__(self, model_path, model_dir=None, feature_type='mel'):
        self.model_path = model_path
        self.model_dir = model_dir
        self.feature_type = feature_type
        self.model = None
        self._resolve_and_load_model()

    def _resolve_and_load_model(self):
        # 1. Resolve filename if model_dir is provided
        if self.model_dir and not os.path.isabs(self.model_path):
            potential_path = os.path.join(self.model_dir, self.model_path)
            if os.path.exists(potential_path):
                self.model_path = potential_path

        # 2. Check if path is a directory (e.g. search for .h5 files)
        if os.path.isdir(self.model_path):
            h5_files = [f for f in os.listdir(self.model_path) if f.endswith('.h5')]
            if h5_files:
                self.model_path = os.path.join(self.model_path, h5_files[0])

        print(f"Loading model from: {self.model_path}")
        self.model = load_model(self.model_path, compile=False)
        print("Model loaded successfully.")

    def _render_audio_in_memory(self, audio_path):
        """Generates a spectrogram image directly in memory from audio track."""
        import librosa
        import librosa.display
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Load audio track
        y, sr = librosa.load(audio_path)
        fig, ax = plt.subplots(nrows=1, sharex=True)
        plt.axis('off')

        if self.feature_type == 'mel':
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
            librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                                     x_axis='time', y_axis='mel', fmax=8000)
        elif self.feature_type == 'mfcc':
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            librosa.display.specshow(mfccs, x_axis='time')
        else:
            plt.close(fig)
            raise ValueError(f"Unknown feature_type: {self.feature_type}")

        # Render canvas to RGB buffer
        fig.canvas.draw()
        img_plot = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        plt.close(fig)

        # Resize to fit CNN inputs
        re_img = cv2.resize(img_plot, (256, 256))
        return re_img

    def predict(self, audio_or_image_path, normalize=True):
        """
        Runs model inference on either an audio track or an image path.
        """
        if not os.path.exists(audio_or_image_path):
            raise FileNotFoundError(f"Input file not found at {audio_or_image_path}")

        ext = os.path.splitext(audio_or_image_path)[1].lower()
        
        # Audio path (WAV, MP3, etc.)
        if ext in ['.wav', '.mp3', '.flac', '.ogg']:
            img = self._render_audio_in_memory(audio_or_image_path)
        # Image path (JPEG, PNG, etc.)
        elif ext in ['.jpeg', '.jpg', '.png', '.bmp']:
            img = cv2.imread(audio_or_image_path)
            if img is None:
                raise ValueError(f"Could not read image file {audio_or_image_path}")
            # Resize
            img = tf.image.resize(img, (256, 256)).numpy()
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        # Normalize pixel intensities
        processed_img = img.astype(np.float32)
        if normalize:
            processed_img = processed_img / 255.0

        # Expand dims for batch inference
        batch_img = np.expand_dims(processed_img, axis=0)

        # Predict
        yhat = self.model.predict(batch_img, verbose=0)
        return yhat

    def get_predicted_class(self, audio_or_image_path, class_names=None, normalize=True):
        """
        Predicts class probabilities and returns the class index, name, and probabilities list.
        """
        yhat = self.predict(audio_or_image_path, normalize=normalize)
        predicted_idx = int(np.argmax(yhat, axis=1)[0])
        
        if class_names and predicted_idx < len(class_names):
            return predicted_idx, class_names[predicted_idx], yhat[0]
        return predicted_idx, None, yhat[0]
