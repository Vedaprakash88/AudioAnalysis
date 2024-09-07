import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import easygui

# get audio data
root = easygui.diropenbox(msg="Select folder with audio files for image conversion", title="Audio Classification")
targetdir = easygui.diropenbox(msg="Select folder to save audio-images", title="Audio Classification")

# create target folders mimicking the root folders

target_folders = os.listdir(root)
root_folders = os.listdir(targetdir)
target_folders.sort(reverse=False)
root_folders.sort(reverse=False)
if target_folders != root_folders:
    folders_to_create = [x for x in target_folders if x not in root_folders]
    for fldr in folders_to_create:
        path = os.path.join(targetdir, fldr)
        os.mkdir(path)

# Walk through ech audio file and save as JPG

for path, subdirs, files in os.walk(root):
    for name in files:
        filename = os.path.join(path, name)
        y, sr = librosa.load(filename)
        target_folder = os.path.join(targetdir, os.path.basename(os.path.normpath(path)))
        fig_name = os.path.join(target_folder, name[:len(name)-4])

        # Visualizing Mel_Spec

        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax = 8000)
        fig, ax = plt.subplots(nrows=1, sharex = True)
        img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                                       x_axis='time', y_axis='mel', fmax=8000)
        plt.axis('off')
        plt.savefig(fig_name + "_Mel_Spec.JPEG", bbox_inches='tight', pad_inches=-0.0001, dpi=1200)
        plt.close()

print("Mel_Specrtrograms saved as images")

