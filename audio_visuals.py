import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

root = "D:\\10. SRH_Academia\\1. All_Notes\\2. Semester 2\\3. Artificial Intelligence\\Project\\DATA\\genres_original"
targetdir = "D:\\10. SRH_Academia\\1. All_Notes\\2. Semester 2\\3. Artificial Intelligence\\Project\\DATA\\Temp\\"

target_folders = os.listdir(root)
root_folders = os.listdir(targetdir)
target_folders.sort(reverse=False)
root_folders.sort(reverse=False)

if target_folders != root_folders:
    folders_to_create = [x for x in target_folders if x not in root_folders]
    for fldr in folders_to_create:
        path = os.path.join(targetdir, fldr)
        os.mkdir(path)
total_files = sum([len(files) for r, d, files in os.walk(root)])

with tqdm(total=total_files, desc="Processing audio files") as pbar:
    for path, subdirs, files in os.walk(root):
        for name in files:
            filename = os.path.join(path, name)
            y, sr = librosa.load(filename)
            target_folder = os.path.join(targetdir, os.path.basename(os.path.normpath(path)))
            fig_name = os.path.join(target_folder, name[:len(name) - 4])

            # waveform
            plt.figure()
            plt.subplot(1, 1, 1)
            librosa.display.waveshow(y, sr=sr, axis='time', color='cyan')
            plt.axis('off')
            plt.savefig(fig_name + "_Wav.JPEG", bbox_inches='tight', pad_inches=-0.1)
            plt.close()

            # Visualizing MFCC

            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            fig, ax = plt.subplots(nrows=1, sharex=True)
            img = librosa.display.specshow(mfccs, x_axis='time')
            plt.axis('off')
            plt.savefig(fig_name + "_MFCC.JPEG", bbox_inches='tight', pad_inches=-0.1)
            plt.close()

            # Visualizing Mel_Spec

            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax = 8000)
            fig, ax = plt.subplots(nrows=1, sharex = True)
            img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                                        x_axis='time', y_axis='mel', fmax=8000)
            plt.axis('off')
            plt.savefig(fig_name + "_Mel_Spec.JPEG", bbox_inches='tight', pad_inches=-0.1)
            plt.close()


            # Spectrogram

            fig, ax = plt.subplots()
            D_highres = librosa.stft(y, hop_length=256, n_fft=4096)
            S_db_hr = librosa.amplitude_to_db(np.abs(D_highres), ref=np.max)
            img = librosa.display.specshow(S_db_hr, hop_length=256, x_axis='time', y_axis='log')
            plt.axis('off')
            plt.savefig(fig_name + "_Spec.JPEG", bbox_inches='tight', pad_inches=-0.1)
            plt.close()

            pbar.update(1)