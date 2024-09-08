import librosa
import numpy as np
import os
import multiprocessing
from functools import partial
from tqdm import tqdm

def parallel_processing(targetdir, num_processes, audio_files):

    process_func = partial(process_audio_file, targetdir=targetdir)

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(process_func, audio_files), total=len(audio_files), desc="Processing audio files"))
    successful = sum([1 for result in results if result is True])
    failed = sum([1 for result in results if result is False])
    print(f"Successfully processed {successful} files.")
    print(f"Failed to process {failed} files.")
    print(f"Total files processed: {successful + failed}")
    print(f"Successfully processed {successful} out of {len(audio_files)} files.")

def process_audio_file(filename, targetdir):
    try:
        y, sr = librosa.load(filename)
        path, name = os.path.split(filename)
        target_folder = os.path.join(targetdir, os.path.basename(os.path.normpath(path)))
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
    except Exception as e:
        print(f'Error processing {filename}: {str(e)}')

def extract_files(root, targetdir, num_processes=None):
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    audio_files = []
    for pathi, _, files in os.walk(root):
        for namei in files:
            if namei.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                audio_files.append(os.path.join(pathi, namei))

    print(f"Found {len(audio_files)} audio files to process.")
    parallel_processing(targetdir, num_processes, audio_files)


if __name__ == "__main__":
    root = "D:\\10. SRH_Academia\\1. All_Notes\\2. Semester 2\\3. Artificial Intelligence\\Project\\DATA\\genres_original"
    targetdir = "D:\\10. SRH_Academia\\1. All_Notes\\2. Semester 2\\3. Artificial Intelligence\\Project\\DATA\\Temp\\"
    extract_files(root, targetdir, num_processes=16)  # Using all 16 threads