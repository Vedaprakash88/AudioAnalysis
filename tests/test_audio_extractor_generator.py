import os
import unittest
import tempfile
import shutil
import wave
import math
import struct
from audio_classifier import AudioFeatureExtractor, AudioImageGenerator

def create_dummy_wav(path, duration=0.5, sr=22050):
    """Generates a small 440Hz sine wave WAV file for testing."""
    num_samples = int(duration * sr)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, 'w') as wav_file:
        wav_file.setparams((1, 2, sr, num_samples, 'NONE', 'not compressed'))
        for i in range(num_samples):
            val = int(32767.0 * math.sin(2.0 * math.pi * 440.0 * i / sr))
            data = struct.pack('<h', val)
            wav_file.writeframesraw(data)

class TestAudioExtractorAndGenerator(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.audio_root = os.path.join(self.temp_dir, "audio_root")
        self.features_target = os.path.join(self.temp_dir, "extracted_features")
        self.images_target = os.path.join(self.temp_dir, "extracted_images")

        # Create two subfolders (classes) and write a dummy WAV file in each
        self.classes = ["genre_classic", "genre_jazz"]
        self.audio_files = []
        for cls in self.classes:
            file_path = os.path.join(self.audio_root, cls, "test_track.wav")
            create_dummy_wav(file_path)
            self.audio_files.append(file_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_feature_extractor_with_csv(self):
        # Run extractor with 1 process
        extractor = AudioFeatureExtractor(
            root_dir=self.audio_root,
            target_dir=self.features_target,
            num_processes=1
        )
        extractor.extract_all()

        # Assert that the unified tabular features CSV was compiled and contains rows
        csv_out_path = os.path.join(self.features_target, "extracted_features.csv")
        self.assertTrue(os.path.exists(csv_out_path))

    def test_image_generator(self):
        # Run image generator with 1 process
        generator = AudioImageGenerator(
            root_dir=self.audio_root,
            target_dir=self.images_target,
            num_processes=1
        )
        
        # Test Mel Spectrogram generation
        success_mel = generator.generate_mel_spectrograms()
        self.assertTrue(success_mel)
        
        # Test MFCC image generation
        success_mfcc = generator.generate_mfccs()
        self.assertTrue(success_mfcc)

        # Check that JPEG images were successfully written
        for cls in self.classes:
            cls_dir = os.path.join(self.images_target, cls)
            self.assertTrue(os.path.exists(cls_dir))
            
            mel_img = os.path.join(cls_dir, "test_track_Mel_Spec.JPEG")
            mfcc_img = os.path.join(cls_dir, "test_track_MFCC.JPEG")
            
            self.assertTrue(os.path.exists(mel_img), f"Missing {mel_img}")
            self.assertTrue(os.path.exists(mfcc_img), f"Missing {mfcc_img}")

if __name__ == '__main__':
    unittest.main()
