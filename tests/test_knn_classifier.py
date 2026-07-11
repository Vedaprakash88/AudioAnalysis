import os
import unittest
import tempfile
import shutil
import pandas as pd
import numpy as np
from audio_analysis import KNNAudioClassifier

class TestKNNAudioClassifier(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.temp_dir, "dummy_features.csv")
        self.plot_path = os.path.join(self.temp_dir, "knn_confusion_matrix.png")
        
        # Generate dummy data with 50 samples (25 per class)
        data = {
            'filename': [f'audio_{i}.wav' for i in range(50)],
            'label': ['genre_A', 'genre_B'] * 25,
            'feat_1': np.random.randn(50),
            'feat_2': np.random.randn(50),
            'feat_3': np.random.randn(50)
        }
        df = pd.DataFrame(data)
        df.to_csv(self.csv_path, index=False)
        self.clf = KNNAudioClassifier(csv_path=self.csv_path, n_neighbors=3)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_pipeline(self):
        # 1. Load and Preprocess
        self.clf.load_and_preprocess()
        self.assertIsNotNone(self.clf.x_train)
        self.assertIsNotNone(self.clf.y_train)
        self.assertEqual(len(self.clf.x_train) + len(self.clf.x_test), 50)
        
        # 2. Train
        self.clf.train()
        self.assertIsNotNone(self.clf.model)
        
        # 3. Evaluate
        results = self.clf.evaluate(save_plot_path=self.plot_path)
        self.assertIn('accuracy', results)
        self.assertIn('confusion_matrix', results)
        
        # Verify the plot file was saved
        self.assertTrue(os.path.exists(self.plot_path))

if __name__ == '__main__':
    unittest.main()
