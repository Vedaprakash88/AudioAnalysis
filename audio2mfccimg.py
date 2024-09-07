import librosa
import matplotlib.pyplot as plt
import numpy as np
import os

root = "D:\\10. SRH_Academia\\1. All_Notes\\2. Semester 2\\3. Artificial Intelligence\\Project\\DATA\\genres_original\\"
targetdir = "D:\\10. SRH_Academia\\1. All_Notes\\2. Semester 2\\3. Artificial Intelligence\\Project\\DATA\\MFCC\\"
target_folders = os.listdir(root)
root_folders = os.listdir(targetdir)
target_folders.sort(reverse=False)
root_folders.sort(reverse=False)
if target_folders != root_folders:
    folders_to_create = [x for x in target_folders if x not in root_folders]
    for fldr in folders_to_create:
        path = os.path.join(targetdir, fldr)
        os.mkdir(path)

for path, subdirs, files in os.walk(root):
    for name in files:
        filename = os.path.join(path, name)
        y, sr = librosa.load(filename)
        target_folder = os.path.join(targetdir, os.path.basename(os.path.normpath(path)))
        fig_name = os.path.join(target_folder, name[:len(name) - 4])

        # Visualizing MFCC

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        fig, ax = plt.subplots(nrows=1, sharex=True)
        img = librosa.display.specshow(mfccs, x_axis='time')
        plt.axis('off')
        plt.savefig(fig_name + "_MFCC.JPEG", bbox_inches='tight', pad_inches=-0.0001, dpi=1200)
        plt.close()
        print(f'{fig_name}.JPEG saved')

print("MFCC saved as images")

