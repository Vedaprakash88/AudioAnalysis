import os
import numpy as np
import pandas as pd
import librosa
import multiprocessing
from functools import partial
from tqdm import tqdm

def _warmup_worker(filename, target_dir):
    """Triggers JIT compilation of all features on a tiny audio slice."""
    try:
        y, sr = librosa.load(filename, duration=0.1)
        D = librosa.stft(y)
        S = np.abs(D)
        librosa.feature.chroma_stft(S=S**2, sr=sr)
        librosa.feature.rms(S=S)
        librosa.feature.spectral_centroid(S=S, sr=sr)
        librosa.feature.spectral_bandwidth(S=S, sr=sr)
        librosa.feature.spectral_rolloff(S=S, sr=sr)
        librosa.feature.zero_crossing_rate(y)
        librosa.decompose.hpss(S)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
    except Exception:
        pass
    return True

def _process_audio_file_worker(filename, target_dir):
    """
    Module-level worker function for parallel processing.
    Extracts raw numpy features and computes 58 DSP features for tabular classification.
    Optimized to reuse precomputed spectrogram matrices and run frequency-domain operations.
    """
    try:
        # Load audio (mono, 22050Hz)
        y, sr = librosa.load(filename, sr=22050)
        path, name = os.path.split(filename)
        subfolder = os.path.basename(os.path.normpath(path))
        target_folder = os.path.join(target_dir, subfolder)
        os.makedirs(target_folder, exist_ok=True)
        base_name = os.path.join(target_folder, os.path.splitext(name)[0])

        # 1. Waveform
        np.save(base_name + "_Waveform.npy", y)

        # 2. Precompute STFT (default n_fft=2048, hop_length=512) to reuse for spectral features
        D = librosa.stft(y)
        S = np.abs(D)
        S_power = S**2

        # 3. Mel Spectrogram (using power spectrogram directly)
        mel_spec = librosa.feature.melspectrogram(S=S_power, sr=sr, n_mels=128, fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        np.save(base_name + "_Mel_Spec.npy", mel_spec_db)

        # 4. MFCC (40 coefficients computed from the Mel Spectrogram - fast!)
        mfccs = librosa.feature.mfcc(S=mel_spec_db, n_mfcc=40)
        np.save(base_name + "_MFCC.npy", mfccs)

        # 5. Spectrogram (STFT high-res, computed only once for saving)
        D_highres = librosa.stft(y, hop_length=256, n_fft=4096)
        S_db_hr = librosa.amplitude_to_db(np.abs(D_highres), ref=np.max)
        np.save(base_name + "_Spec.npy", S_db_hr)

        # 6. Extract Tabular DSP Features
        feat_dict = {
            'filename': name,
            'length': len(y)
        }

        # Chroma STFT (using power spectrogram)
        chroma = librosa.feature.chroma_stft(S=S_power, sr=sr)
        feat_dict['chroma_stft_mean'] = float(np.mean(chroma))
        feat_dict['chroma_stft_var'] = float(np.var(chroma))

        # RMS (using magnitude spectrogram)
        rms = librosa.feature.rms(S=S)
        feat_dict['rms_mean'] = float(np.mean(rms))
        feat_dict['rms_var'] = float(np.var(rms))

        # Spectral Centroid (using magnitude spectrogram)
        spec_cent = librosa.feature.spectral_centroid(S=S, sr=sr)
        feat_dict['spectral_centroid_mean'] = float(np.mean(spec_cent))
        feat_dict['spectral_centroid_var'] = float(np.var(spec_cent))

        # Spectral Bandwidth (using magnitude spectrogram)
        spec_bw = librosa.feature.spectral_bandwidth(S=S, sr=sr)
        feat_dict['spectral_bandwidth_mean'] = float(np.mean(spec_bw))
        feat_dict['spectral_bandwidth_var'] = float(np.var(spec_bw))

        # Rolloff (using magnitude spectrogram)
        rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)
        feat_dict['rolloff_mean'] = float(np.mean(rolloff))
        feat_dict['rolloff_var'] = float(np.var(rolloff))

        # Zero Crossing Rate (runs in milliseconds on time domain)
        zcr = librosa.feature.zero_crossing_rate(y)
        feat_dict['zero_crossing_rate_mean'] = float(np.mean(zcr))
        feat_dict['zero_crossing_rate_var'] = float(np.var(zcr))

        # Frequency-domain HPSS (15x speedup compared to time-domain hpss)
        harmonic_mag, percussive_mag = librosa.decompose.hpss(S)
        feat_dict['harmony_mean'] = float(np.mean(harmonic_mag))
        feat_dict['harmony_var'] = float(np.var(harmonic_mag))
        feat_dict['perceptr_mean'] = float(np.mean(percussive_mag))
        feat_dict['perceptr_var'] = float(np.var(percussive_mag))

        # Optimized Tempo
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
        if isinstance(tempo, np.ndarray):
            feat_dict['tempo'] = float(tempo[0]) if len(tempo) > 0 else 120.0
        else:
            feat_dict['tempo'] = float(tempo)

        # 20 MFCC stats (taking first 20 coefficients of our 40-MFCC calculation)
        for i in range(20):
            feat_dict[f'mfcc{i+1}_mean'] = float(np.mean(mfccs[i]))
            feat_dict[f'mfcc{i+1}_var'] = float(np.var(mfccs[i]))

        feat_dict['label'] = subfolder
        return feat_dict
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        return None

class AudioFeatureExtractor:
    """
    Extracts raw numpy features and aggregates tabular DSP properties
    across all files in a root directory, saving them in a unified CSV.
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

        # 1. JIT Compilation Warm-Up Phase
        print(f"Initializing process pool and warming up JIT compilers ({self.num_processes} workers)...")
        warmup_func = partial(_warmup_worker, target_dir=self.target_dir)
        with multiprocessing.Pool(processes=self.num_processes) as pool:
            # Map a fast 0.1s slice task to all workers to warm them up
            pool.map(warmup_func, [audio_files[0]] * self.num_processes)
        print("Warm up complete. Processing dataset...")

        # 2. Main Extraction Run
        process_func = partial(_process_audio_file_worker, target_dir=self.target_dir)
        with multiprocessing.Pool(processes=self.num_processes) as pool:
            results = list(tqdm(pool.imap(process_func, audio_files), total=len(audio_files), desc="Extracting features"))

        # Filter out failed runs
        valid_features = [res for res in results if res is not None]

        # Compile features into tabular CSV
        if valid_features:
            df = pd.DataFrame(valid_features)
            csv_out_path = os.path.join(self.target_dir, "extracted_features.csv")
            df.to_csv(csv_out_path, index=False)
            print(f"Successfully compiled feature table to: {csv_out_path}")

        successful = len(valid_features)
        failed = len(audio_files) - successful
        print(f"\nSuccessfully processed {successful} files.")
        print(f"Failed to process {failed} files.")
        print(f"Total files processed: {successful + failed} / {len(audio_files)}")
