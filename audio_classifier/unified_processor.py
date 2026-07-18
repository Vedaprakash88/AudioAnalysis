import os
import sys
import numpy as np
import pandas as pd
import librosa
import librosa.display
import multiprocessing
from functools import partial
from tqdm import tqdm

# Ensure matplotlib runs in headless mode
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _unified_processing_worker(args):
    """
    Multiprocessing worker function that loads an audio file exactly once
    and extracts/generates the requested representations.
    """
    (filename, run_features, run_mel, run_mfcc,
     features_target_dir, melspec_target_dir, mfcc_target_dir) = args

    try:
        # Load audio (mono, 22050Hz) - identical settings for both extraction and image generation
        y, sr = librosa.load(filename, sr=22050)
        path, name = os.path.split(filename)
        subfolder = os.path.basename(os.path.normpath(path))

        feat_dict = None
        mel_spec = None
        mel_spec_db = None
        S_power = None

        # Precompute STFT & power spectrogram if needed by features or mel image
        if run_features or run_mel:
            D = librosa.stft(y)
            S = np.abs(D)
            S_power = S**2

        # 1. Tabular features extraction
        if run_features:
            mel_spec = librosa.feature.melspectrogram(S=S_power, sr=sr, n_mels=128, fmax=8000)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            mfccs = librosa.feature.mfcc(S=mel_spec_db, n_mfcc=40)

            feat_dict = {
                'filename': name,
                'length': len(y)
            }

            # Chroma STFT
            chroma = librosa.feature.chroma_stft(S=S_power, sr=sr)
            feat_dict['chroma_stft_mean'] = float(np.mean(chroma))
            feat_dict['chroma_stft_var'] = float(np.var(chroma))

            # RMS
            rms = librosa.feature.rms(S=S)
            feat_dict['rms_mean'] = float(np.mean(rms))
            feat_dict['rms_var'] = float(np.var(rms))

            # Spectral Centroid
            spec_cent = librosa.feature.spectral_centroid(S=S, sr=sr)
            feat_dict['spectral_centroid_mean'] = float(np.mean(spec_cent))
            feat_dict['spectral_centroid_var'] = float(np.var(spec_cent))

            # Spectral Bandwidth
            spec_bw = librosa.feature.spectral_bandwidth(S=S, sr=sr)
            feat_dict['spectral_bandwidth_mean'] = float(np.mean(spec_bw))
            feat_dict['spectral_bandwidth_var'] = float(np.var(spec_bw))

            # Rolloff
            rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)
            feat_dict['rolloff_mean'] = float(np.mean(rolloff))
            feat_dict['rolloff_var'] = float(np.var(rolloff))

            # Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(y)
            feat_dict['zero_crossing_rate_mean'] = float(np.mean(zcr))
            feat_dict['zero_crossing_rate_var'] = float(np.var(zcr))

            # HPSS
            harmonic_mag, percussive_mag = librosa.decompose.hpss(S)
            feat_dict['harmony_mean'] = float(np.mean(harmonic_mag))
            feat_dict['harmony_var'] = float(np.var(harmonic_mag))
            feat_dict['perceptr_mean'] = float(np.mean(percussive_mag))
            feat_dict['perceptr_var'] = float(np.var(percussive_mag))

            # Tempo
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
            if isinstance(tempo, np.ndarray):
                feat_dict['tempo'] = float(tempo[0]) if len(tempo) > 0 else 120.0
            else:
                feat_dict['tempo'] = float(tempo)

            # 20 MFCC stats
            for i in range(20):
                feat_dict[f'mfcc{i+1}_mean'] = float(np.mean(mfccs[i]))
                feat_dict[f'mfcc{i+1}_var'] = float(np.var(mfccs[i]))

            feat_dict['label'] = subfolder

        # 2. Mel Spectrogram image generation
        if run_mel:
            # Use precalculated power spectrogram if available to avoid redundant computation
            if mel_spec is None:
                mel_spec = librosa.feature.melspectrogram(S=S_power, sr=sr, n_mels=128, fmax=8000)
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            target_folder = os.path.join(melspec_target_dir, subfolder)
            os.makedirs(target_folder, exist_ok=True)
            fig_name = os.path.join(target_folder, os.path.splitext(name)[0])

            fig, ax = plt.subplots(nrows=1, sharex=True)
            librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel', fmax=8000)
            plt.axis('off')
            plt.savefig(fig_name + "_Mel_Spec.JPEG", bbox_inches='tight', pad_inches=-0.0001, dpi=1200)
            plt.close(fig)

        # 3. MFCC image generation
        if run_mfcc:
            # Always computed directly with 13 coefficients as per original implementation
            mfccs_img = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

            target_folder = os.path.join(mfcc_target_dir, subfolder)
            os.makedirs(target_folder, exist_ok=True)
            fig_name = os.path.join(target_folder, os.path.splitext(name)[0])

            fig, ax = plt.subplots(nrows=1, sharex=True)
            librosa.display.specshow(mfccs_img, x_axis='time')
            plt.axis('off')
            plt.savefig(fig_name + "_MFCC.JPEG", bbox_inches='tight', pad_inches=-0.0001, dpi=1200)
            plt.close(fig)

        return {'status': 'success', 'features': feat_dict, 'file': name}
    except Exception as e:
        print(f"Error processing audio file {filename}: {str(e)}")
        return {'status': 'failed', 'file': filename, 'error': str(e)}

class UnifiedAudioProcessor:
    """
    Coordinates unified, parallelized audio feature extraction and image generation
    to load each audio file only once.
    """
    def __init__(self, root_dir, features_target_dir=None, melspec_target_dir=None, mfcc_target_dir=None, num_processes=None):
        self.root_dir = root_dir
        self.features_target_dir = features_target_dir
        self.melspec_target_dir = melspec_target_dir
        self.mfcc_target_dir = mfcc_target_dir
        self.num_processes = num_processes if num_processes is not None else multiprocessing.cpu_count()

    def _get_audio_files(self):
        audio_files = []
        for path, _, files in os.walk(self.root_dir):
            for file in files:
                if file.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                    audio_files.append(os.path.join(path, file))
        return audio_files

    def _prepare_folders(self, target_dir):
        if not target_dir:
            return False
        try:
            target_folders = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
            os.makedirs(target_dir, exist_ok=True)
            root_folders = os.listdir(target_dir)
            for fldr in target_folders:
                if fldr not in root_folders:
                    os.makedirs(os.path.join(target_dir, fldr), exist_ok=True)
            return True
        except Exception as e:
            print(f"Error preparing directory {target_dir}: {e}")
            return False

    def run_pipeline(self, run_features=True, run_mel=True, run_mfcc=True):
        """
        Executes the combined preprocessing pipeline in parallel.
        """
        audio_files = self._get_audio_files()
        print(f"Found {len(audio_files)} audio files to process.")
        if not audio_files:
            return True

        # Pre-create output folders to prevent race conditions in worker threads
        if run_features and self.features_target_dir:
            os.makedirs(self.features_target_dir, exist_ok=True)
        if run_mel and self.melspec_target_dir:
            self._prepare_folders(self.melspec_target_dir)
        if run_mfcc and self.mfcc_target_dir:
            self._prepare_folders(self.mfcc_target_dir)

        # Build execution arguments for workers
        worker_args = []
        for f in audio_files:
            worker_args.append((
                f,
                run_features,
                run_mel,
                run_mfcc,
                self.features_target_dir,
                self.melspec_target_dir,
                self.mfcc_target_dir
            ))

        print(f"Processing dataset in parallel using {self.num_processes} processes...")
        
        with multiprocessing.Pool(processes=self.num_processes) as pool:
            results = list(tqdm(
                pool.imap(_unified_processing_worker, worker_args),
                total=len(audio_files),
                desc="Preprocessing audio"
            ))

        # Separate feature dictionary compilation and count success
        valid_features = []
        success_count = 0
        for res in results:
            if res and res.get('status') == 'success':
                success_count += 1
                if run_features and res.get('features') is not None:
                    valid_features.append(res['features'])

        # Compile CSV if tabular features were extracted
        if run_features and valid_features and self.features_target_dir:
            df = pd.DataFrame(valid_features)
            csv_out_path = os.path.join(self.features_target_dir, "extracted_features.csv")
            df.to_csv(csv_out_path, index=False)
            print(f"Successfully compiled feature table to: {csv_out_path}")

        failed_count = len(audio_files) - success_count
        print(f"\nPreprocessing Summary:")
        print(f"  Successfully processed: {success_count} files")
        print(f"  Failed: {failed_count} files")
        print(f"  Total: {len(audio_files)}")

        return success_count == len(audio_files)
