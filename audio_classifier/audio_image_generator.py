import os
import multiprocessing
import librosa
import librosa.display
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from functools import partial
from tqdm import tqdm

def _process_mel_file_worker(filename, target_dir):
    """
    Module-level worker to process Mel Spectrogram image generation in parallel.
    """
    try:
        y, sr = librosa.load(filename)
        base_folder = os.path.basename(os.path.normpath(os.path.dirname(filename)))
        target_folder = os.path.join(target_dir, base_folder)
        os.makedirs(target_folder, exist_ok=True)
        fig_name = os.path.join(target_folder, os.path.splitext(os.path.basename(filename))[0])

        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        fig, ax = plt.subplots(nrows=1, sharex=True)
        librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                                 x_axis='time', y_axis='mel', fmax=8000)
        plt.axis('off')
        plt.savefig(fig_name + "_Mel_Spec.JPEG", bbox_inches='tight', pad_inches=-0.0001, dpi=1200)
        plt.close()
        return True
    except Exception as e:
        print(f"Error generating Mel Spectrogram image for {filename}: {e}")
        return False

def _process_mfcc_file_worker(filename, target_dir):
    """
    Module-level worker to process MFCC image generation in parallel.
    """
    try:
        y, sr = librosa.load(filename)
        base_folder = os.path.basename(os.path.normpath(os.path.dirname(filename)))
        target_folder = os.path.join(target_dir, base_folder)
        os.makedirs(target_folder, exist_ok=True)
        fig_name = os.path.join(target_folder, os.path.splitext(os.path.basename(filename))[0])

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        fig, ax = plt.subplots(nrows=1, sharex=True)
        librosa.display.specshow(mfccs, x_axis='time')
        plt.axis('off')
        plt.savefig(fig_name + "_MFCC.JPEG", bbox_inches='tight', pad_inches=-0.0001, dpi=1200)
        plt.close()
        return True
    except Exception as e:
        print(f"Error generating MFCC image for {filename}: {e}")
        return False

from .unified_processor import UnifiedAudioProcessor

class AudioImageGenerator:
    """
    Generates Mel Spectrogram and MFCC images from audio files and saves them as JPEGs.
    Delegates to UnifiedAudioProcessor to load each file exactly once.
    """
    def __init__(self, root_dir, target_dir, num_processes=None):
        self.root_dir = root_dir
        self.target_dir = target_dir
        self.num_processes = num_processes

    def generate_mel_spectrograms(self):
        processor = UnifiedAudioProcessor(
            root_dir=self.root_dir,
            melspec_target_dir=self.target_dir,
            num_processes=self.num_processes
        )
        return processor.run_pipeline(run_features=False, run_mel=True, run_mfcc=False)

    def generate_mfccs(self):
        processor = UnifiedAudioProcessor(
            root_dir=self.root_dir,
            mfcc_target_dir=self.target_dir,
            num_processes=self.num_processes
        )
        return processor.run_pipeline(run_features=False, run_mel=False, run_mfcc=True)
