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
    Inference helper to predict class probabilities from audio spectrogram images.
    """
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            # Try to see if it's a directory containing models
            if os.path.isdir(self.model_path):
                # Search for .h5 or saved_model.pb
                h5_files = [f for f in os.listdir(self.model_path) if f.endswith('.h5')]
                if h5_files:
                    self.model_path = os.path.join(self.model_path, h5_files[0])
                else:
                    # Let tf load from directory (SavedModel format)
                    pass
            
        print(f"Loading model from: {self.model_path}")
        self.model = load_model(self.model_path, compile=False)
        print("Model loaded successfully.")

    def predict(self, image_path, normalize=True):
        """
        Loads an image, processes it, and returns the raw model predictions.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image file {image_path}")

        # Resize to match CNN input shape
        re_img = tf.image.resize(img, (256, 256))
        
        # Convert to numpy and normalize
        processed_img = re_img.numpy()
        if normalize:
            processed_img = processed_img / 255.0

        # Expand dims for batch size of 1
        batch_img = np.expand_dims(processed_img, axis=0)

        # Run inference
        yhat = self.model.predict(batch_img, verbose=0)
        return yhat

    def get_predicted_class(self, image_path, class_names=None, normalize=True):
        """
        Predicts class probabilities and returns the class index and name (if class_names provided).
        """
        yhat = self.predict(image_path, normalize=normalize)
        predicted_idx = int(np.argmax(yhat, axis=1)[0])
        
        if class_names and predicted_idx < len(class_names):
            return predicted_idx, class_names[predicted_idx], yhat[0]
        return predicted_idx, None, yhat[0]
