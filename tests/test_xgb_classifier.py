import os
import unittest
import tempfile
import shutil
import pandas as pd
import numpy as np
from xgb_feature_classifier import XGBFeatureClassifier

class TestXGBFeatureClassifier(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.temp_dir, "dummy_features.csv")
        self.plot_path = os.path.join(self.temp_dir, "xgb_importance.png")
        
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

        # Set small parameters for test speed
        self.model_params = {
            'max_depth': 2,
            'learning_rate': 0.1,
            'n_estimators': 5,
            'verbosity': 0,
            'booster': 'gbtree',
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'multi:softmax',
            'eval_metric': 'logloss',
            'nthread': 1
        }
        self.clf = XGBFeatureClassifier(csv_path=self.csv_path, model_params=self.model_params)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_pipeline(self):
        # 1. Load and Preprocess
        self.clf.load_and_preprocess()
        self.assertIsNotNone(self.clf.x_train)
        self.assertIsNotNone(self.clf.y_train)
        self.assertEqual(len(self.clf.x_train) + len(self.clf.x_test), 50)
        
        # 2. Train
        self.clf.train(verbose=False)
        self.assertIsNotNone(self.clf.model)
        
        # 3. Evaluate with 10-fold cross-validation (works now that class has 25 members)
        results = self.clf.evaluate(run_cv=True)
        self.assertIn('accuracy', results)
        self.assertIn('cv_accuracy_mean', results)
        
        # 4. Feature Importance Plot
        self.clf.plot_feature_importance(output_path=self.plot_path)
        self.assertTrue(os.path.exists(self.plot_path))

if __name__ == '__main__':
    unittest.main()
