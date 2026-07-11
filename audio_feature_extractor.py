import os
import numpy as np
import librosa
import multiprocessing
from functools import partial
from tqdm import tqdm

def _process_audio_file_worker(filename, target_dir):
    """
    Module-level worker function for parallel processing.
    Must be at module level for Windows multiprocessing compatibility.
    """
    try:
        y, sr = librosa.load(filename)
        path, name = os.path.split(filename)
        # Identify subfolder name (e.g. class label)
        subfolder = os.path.basename(os.path.normpath(path))
        target_folder = os.path.join(target_dir, subfolder)
        os.makedirs(target_folder, exist_ok=True)
        base_name = os.path.join(target_folder, os.path.splitext(name)[0])

        # Waveform
        np.save(base_name + "_Waveform.npy", y)

        # MFCC
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        np.save(base_name + "_MFCC.npy", mfccs)

        # Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        np.save(base_name + "_Mel_Spec.npy", mel_spec_db)

        # Spectrogram
        D_highres = librosa.stft(y, hop_length=256, n_fft=4096)
        S_db_hr = librosa.amplitude_to_db(np.abs(D_highres), ref=np.max)
        np.save(base_name + "_Spec.npy", S_db_hr)
        return True
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        return False

class AudioFeatureExtractor:
    """
    Extracts raw numpy features (Waveform, MFCC, Mel Spectrogram, STFT Spectrogram)
    from audio files and saves them as .npy files.
    """
    def __init__(self, root_dir, target_dir, num_processes=None):
        self.root_dir = root_dir
        self.target_dir = target_dir
        self.num_processes = num_processes if num_processes is not None else multiprocessing.cpu_count()

    def extract_all(self):
        audio_files = []
        for path, _, files in os.walk(self.root_dir):
            for file in files:
                if file.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                    audio_files.append(os.path.join(path, file))

        print(f"Found {len(audio_files)} audio files to process.")
        if not audio_files:
            return

        os.makedirs(self.target_dir, exist_ok=True)

        process_func = partial(_process_audio_file_worker, target_dir=self.target_dir)

        with multiprocessing.Pool(processes=self.num_processes) as pool:
            results = list(tqdm(pool.imap(process_func, audio_files), total=len(audio_files), desc="Extracting features"))

        successful = sum([1 for r in results if r is True])
        failed = sum([1 for r in results if r is False])
        print(f"\nSuccessfully processed {successful} files.")
        print(f"Failed to process {failed} files.")
        print(f"Total files processed: {successful + failed} / {len(audio_files)}")
