import os
import unittest
import tempfile
import shutil
import cv2
import numpy as np
from audio_cnn_classifier import AudioCNNClassifier
from audio_cnn_predictor import AudioCNNPredictor

def create_dummy_jpeg(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Generate 256x256 random image
    img = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
    cv2.imwrite(path, img)

class TestAudioCNNPipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, "data")
        self.model_dir = os.path.join(self.temp_dir, "model")
        self.log_dir = os.path.join(self.temp_dir, "logs")
        self.checkpoint_dir = os.path.join(self.temp_dir, "checkpoints")

        # Create dummy images (10 for each class, total 20 images)
        self.classes = ["genre_A", "genre_B"]
        for cls in self.classes:
            for i in range(10):
                img_path = os.path.join(self.data_dir, cls, f"img_{i}.jpeg")
                create_dummy_jpeg(img_path)

        self.classifier = AudioCNNClassifier(
            data_dir=self.data_dir,
            model_dir=self.model_dir,
            log_dir=self.log_dir,
            checkpoint_dir=self.checkpoint_dir,
            batch_size=2,
            epochs=1  # Only 1 epoch for quick test run
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_cnn_pipeline(self):
        # 1. Prepare Data
        self.classifier.prepare_data()
        self.assertEqual(len(self.classifier.class_names), 2)
        
        # 2. Train
        # Save plots is disabled or set to True, let's keep it True
        self.classifier.train(save_plots=True)
        self.assertIsNotNone(self.classifier.model)
        
        # Verify plots were saved
        self.assertTrue(os.path.exists(os.path.join(self.model_dir, "cnn_loss_curves.png")))
        self.assertTrue(os.path.exists(os.path.join(self.model_dir, "cnn_accuracy_curves.png")))

        # 3. Evaluate
        acc = self.classifier.evaluate()
        self.assertIsNotNone(acc)
        self.assertTrue(0.0 <= acc <= 1.0)

        # 4. Save Model
        model_name = "test_model.h5"
        model_path = self.classifier.save_model(model_name)
        self.assertTrue(os.path.exists(model_path))

        # 5. Predictor Inference
        predictor = AudioCNNPredictor(model_path=model_path)
        # Test basic prediction
        test_img = os.path.join(self.data_dir, "genre_A", "img_0.jpeg")
        yhat = predictor.predict(test_img)
        self.assertEqual(yhat.shape, (1, 2))
        
        # Test class prediction
        idx, name, probs = predictor.get_predicted_class(test_img, class_names=self.classifier.class_names)
        self.assertIn(idx, [0, 1])
        self.assertEqual(name, self.classifier.class_names[idx])
        self.assertEqual(len(probs), 2)

if __name__ == '__main__':
    unittest.main()
