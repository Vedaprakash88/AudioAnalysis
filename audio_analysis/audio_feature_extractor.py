import os
import numpy as np
import pandas as pd
import librosa
import multiprocessing
from functools import partial
from tqdm import tqdm

def _extract_raw_arrays_worker(filename, target_dir):
    """
    Worker to load raw audio and save the base representations (.npy files).
    Saves Waveform, Mel Spec, MFCC, and high-resolution STFT spectrogram.
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

        # 2. Precompute STFT once for Mel and MFCC
        D = librosa.stft(y)
        S = np.abs(D)
        S_power = S**2

        # 3. Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(S=S_power, sr=sr, n_mels=128, fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        np.save(base_name + "_Mel_Spec.npy", mel_spec_db)

        # 4. MFCC (40 coefficients)
        mfccs = librosa.feature.mfcc(S=mel_spec_db, n_mfcc=40)
        np.save(base_name + "_MFCC.npy", mfccs)

        # 5. Spectrogram (STFT high-res, 4096 bins, hop size 256)
        D_highres = librosa.stft(y, hop_length=256, n_fft=4096)
        S_db_hr = librosa.amplitude_to_db(np.abs(D_highres), ref=np.max)
        np.save(base_name + "_Spec.npy", S_db_hr)

        return True
    except Exception as e:
        print(f"Error extracting raw arrays for {filename}: {str(e)}")
        return False

def _extract_tabular_features_worker(filename, target_dir):
    """
    Worker to load pre-extracted numpy arrays and compute tabular properties.
    Extremely fast because it loads binary arrays instead of reloading/decoding audio files.
    """
    try:
        path, name = os.path.split(filename)
        subfolder = os.path.basename(os.path.normpath(path))
        target_folder = os.path.join(target_dir, subfolder)
        base_name = os.path.join(target_folder, os.path.splitext(name)[0])

        # Load pre-saved numpy arrays
        y = np.load(base_name + "_Waveform.npy")
        mel_spec_db = np.load(base_name + "_Mel_Spec.npy")
        mfccs = np.load(base_name + "_MFCC.npy")

        sr = 22050
        D = librosa.stft(y)
        S = np.abs(D)
        S_power = S**2

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

        # Freq-domain HPSS
        harmonic_mag, percussive_mag = librosa.decompose.hpss(S)
        feat_dict['harmony_mean'] = float(np.mean(harmonic_mag))
        feat_dict['harmony_var'] = float(np.var(harmonic_mag))
        feat_dict['perceptr_mean'] = float(np.mean(percussive_mag))
        feat_dict['perceptr_var'] = float(np.var(percussive_mag))

        # Optimized Tempo
        # Re-use mel_spec_db to extract tempo strength envelope for speed
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
        return feat_dict
    except Exception as e:
        print(f"Error extracting tabular features for {filename}: {str(e)}")
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

        # 1. Step 1: Extract raw representation arrays (.npy)
        print("\n--- Step 1/2: Extracting raw representation arrays (.npy) ---")
        raw_func = partial(_extract_raw_arrays_worker, target_dir=self.target_dir)
        with multiprocessing.Pool(processes=self.num_processes) as pool:
            raw_results = list(tqdm(pool.imap_unordered(raw_func, audio_files),
                                    total=len(audio_files),
                                    desc="Extracting .npy files"))
        
        successful_raw = sum(raw_results)
        print(f"Completed raw extraction: {successful_raw} succeeded, {len(audio_files) - successful_raw} failed.")

        # Filter audio files to only successfully extracted ones for tabular pass
        valid_files = [audio_files[i] for i, success in enumerate(raw_results) if success]

        # 2. Step 2: Compute tabular stats
        print("\n--- Step 2/2: Computing tabular stats for CSV compilation ---")
        tab_func = partial(_extract_tabular_features_worker, target_dir=self.target_dir)
        with multiprocessing.Pool(processes=self.num_processes) as pool:
            tab_results = list(tqdm(pool.imap(tab_func, valid_files),
                                    total=len(valid_files),
                                    desc="Compiling tabular features"))

        # Filter out failed runs
        valid_features = [res for res in tab_results if res is not None]

        # Compile features into tabular CSV
        if valid_features:
            df = pd.DataFrame(valid_features)
            csv_out_path = os.path.join(self.target_dir, "extracted_features.csv")
            df.to_csv(csv_out_path, index=False)
            print(f"Successfully compiled feature table to: {csv_out_path}")

        successful = len(valid_features)
        failed = len(audio_files) - successful
        print(f"\nSummary:")
        print(f"  Successfully processed: {successful} files")
        print(f"  Failed: {failed} files")
        print(f"  Total: {successful + failed} / {len(audio_files)}")
