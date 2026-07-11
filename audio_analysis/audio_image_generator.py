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

class AudioImageGenerator:
    """
    Generates Mel Spectrogram and MFCC images from audio files and saves them as JPEGs.
    """
    def __init__(self, root_dir, target_dir, num_processes=None):
        self.root_dir = root_dir
        self.target_dir = target_dir
        self.num_processes = num_processes if num_processes is not None else multiprocessing.cpu_count()

    def _get_audio_files(self):
        audio_files = []
        for path, _, files in os.walk(self.root_dir):
            for file in files:
                if file.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                    audio_files.append(os.path.join(path, file))
        return audio_files

    def _prepare_folders(self):
        try:
            target_folders = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
            os.makedirs(self.target_dir, exist_ok=True)
            root_folders = os.listdir(self.target_dir)
            for fldr in target_folders:
                if fldr not in root_folders:
                    os.makedirs(os.path.join(self.target_dir, fldr), exist_ok=True)
            return True
        except Exception as e:
            print(f"Error preparing folders: {e}")
            return False

    def generate_mel_spectrograms(self):
        if not self._prepare_folders():
            return False

        audio_files = self._get_audio_files()
        print(f"Found {len(audio_files)} audio files for Mel Spectrogram conversion.")
        if not audio_files:
            return True

        process_with_target = partial(_process_mel_file_worker, target_dir=self.target_dir)

        with multiprocessing.Pool(processes=self.num_processes) as pool:
            results = list(tqdm(pool.imap_unordered(process_with_target, audio_files),
                                total=len(audio_files),
                                desc="Generating Mel Spectrogram images"))

        successful_count = sum(results)
        print(f"Processed {len(audio_files)} files. {successful_count} successful, {len(audio_files) - successful_count} failed.")
        return successful_count == len(audio_files)

    def generate_mfccs(self):
        if not self._prepare_folders():
            return False

        audio_files = self._get_audio_files()
        print(f"Found {len(audio_files)} audio files for MFCC conversion.")
        if not audio_files:
            return True

        process_with_target = partial(_process_mfcc_file_worker, target_dir=self.target_dir)

        with multiprocessing.Pool(processes=self.num_processes) as pool:
            results = list(tqdm(pool.imap_unordered(process_with_target, audio_files),
                                total=len(audio_files),
                                desc="Generating MFCC images"))

        successful_count = sum(results)
        print(f"Processed {len(audio_files)} files. {successful_count} successful, {len(audio_files) - successful_count} failed.")
        return successful_count == len(audio_files)
