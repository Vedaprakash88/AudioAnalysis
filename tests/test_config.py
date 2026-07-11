import os
import unittest
import tempfile
import shutil
from audio_analysis import load_config

class TestConfigLoader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.ini_path = os.path.join(self.temp_dir, "test_config.ini")
        
        # Write dummy ini file
        ini_content = """[paths]
audio_root_dir = /dummy/audio/root
output_dir = /dummy/out/dir

[cnn]
feature_type = mfcc
batch_size = 5
epochs = 20
model_name = my_test_model.h5

[knn]
n_neighbors = 8

[xgb]
n_estimators = 100
learning_rate = 0.05
max_depth = 4
"""
        with open(self.ini_path, 'w') as f:
            f.write(ini_content)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_load_config(self):
        config = load_config(self.ini_path)
        self.assertEqual(config['audio_root_dir'], "/dummy/audio/root")
        self.assertEqual(config['cnn_feature_type'], "mfcc")
        self.assertEqual(config['cnn_batch_size'], 5)
        self.assertEqual(config['cnn_epochs'], 20)
        self.assertEqual(config['cnn_model_name'], "my_test_model.h5")
        self.assertEqual(config['knn_n_neighbors'], 8)
        self.assertEqual(config['xgb_params']['n_estimators'], 100)
        self.assertEqual(config['xgb_params']['learning_rate'], 0.05)
        self.assertEqual(config['xgb_params']['max_depth'], 4)

if __name__ == '__main__':
    unittest.main()
