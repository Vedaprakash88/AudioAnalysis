import os
import numpy as np
import pandas as pd
import librosa
import multiprocessing
from functools import partial
from tqdm import tqdm

def _extract_tabular_features_direct_worker(filename):
    """
    Worker to load raw audio and extract tabular features directly using librosa.
    Bypasses saving/loading intermediate .npy files to disk.
    """
    try:
        # Load audio (mono, 22050Hz)
        y, sr = librosa.load(filename, sr=22050)
        path, name = os.path.split(filename)
        subfolder = os.path.basename(os.path.normpath(path))

        # Precompute STFT
        D = librosa.stft(y)
        S = np.abs(D)
        S_power = S**2

        # Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(S=S_power, sr=sr, n_mels=128, fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # MFCC (40 coefficients)
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

from .unified_processor import UnifiedAudioProcessor

class AudioFeatureExtractor:
    """
    Extracts raw features and aggregates tabular DSP properties
    across all files in a root directory directly, saving them in a unified CSV.
    Delegates to UnifiedAudioProcessor to load each file exactly once.
    """
    def __init__(self, root_dir, target_dir, num_processes=None):
        self.root_dir = root_dir
        self.target_dir = target_dir
        self.num_processes = num_processes

    def extract_all(self):
        processor = UnifiedAudioProcessor(
            root_dir=self.root_dir,
            features_target_dir=self.target_dir,
            num_processes=self.num_processes
        )
        processor.run_pipeline(run_features=True, run_mel=False, run_mfcc=False)
