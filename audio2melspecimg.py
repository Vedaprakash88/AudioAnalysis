import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import easygui
import multiprocessing
from functools import partial

def process_audio_file(filename, targetdir):
    try:
        y, sr = librosa.load(filename)
        base_folder = os.path.basename(os.path.normpath(os.path.dirname(filename)))
        target_folder = os.path.join(targetdir, base_folder)
        os.makedirs(target_folder, exist_ok=True)
        fig_name = os.path.join(target_folder, os.path.splitext(os.path.basename(filename))[0])

        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        fig, ax = plt.subplots(nrows=1, sharex=True)
        librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                                 x_axis='time', y_axis='mel', fmax=8000)  # No need to store the image in a variable
        plt.axis('off')
        plt.savefig(fig_name + "_Mel_Spec.JPEG", bbox_inches='tight', pad_inches=-0.0001, dpi=1200)
        plt.close()
        return True
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return False

def main(root, targetdir, num_processes=None):
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    all_files = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            filename = os.path.join(path, name)
            all_files.append(filename)

    pool = multiprocessing.Pool(processes=num_processes)
    process_with_target = partial(process_audio_file, targetdir=targetdir)
    results = pool.map(process_with_target, all_files)
    pool.close()
    pool.join()

    successful_count = sum(results)
    print(f"Processed {len(all_files)} files. {successful_count} successful, {len(all_files) - successful_count} failed.")
    print("Mel_Spectrograms saved as images") #Moved here


if __name__ == "__main__":
    root = easygui.diropenbox(msg="Select folder with audio files for image conversion", title="Audio Classification")
    targetdir = easygui.diropenbox(msg="Select folder to save audio-images", title="Audio Classification")

    # create target folders mimicking the root folders (your original folder creation logic)
    target_folders = os.listdir(root)
    root_folders = os.listdir(targetdir)  # Check existing folders in target dir
    target_folders.sort()  # Sort in place
    root_folders.sort()
    if target_folders != root_folders:
        folders_to_create = [x for x in target_folders if x not in root_folders]
        for fldr in folders_to_create:
            path = os.path.join(targetdir, fldr)
            os.makedirs(path, exist_ok=True) # Use makedirs to avoid error if exists


    main(root, targetdir)