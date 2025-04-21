import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import multiprocessing
from functools import partial

def process_audio_file(filename, targetdir):
    try:
        y, sr = librosa.load(filename)
        base_folder = os.path.basename(os.path.normpath(os.path.dirname(filename)))
        target_folder = os.path.join(targetdir, base_folder)
        os.makedirs(target_folder, exist_ok=True)
        fig_name = os.path.join(target_folder, os.path.splitext(os.path.basename(filename))[0])

        # Visualizing MFCC

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        fig, ax = plt.subplots(nrows=1, sharex=True)
        img = librosa.display.specshow(mfccs, x_axis='time')
        plt.axis('off')
        plt.savefig(fig_name + "_MFCC.JPEG", bbox_inches='tight', pad_inches=-0.0001, dpi=1200)
        plt.close()
        return True
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return False

def multi_proc_inititation(root, targetdir, num_processes=multiprocessing.cpu_count()):
    try:
        all_files = []
        for path, _, files in os.walk(root):
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
    except Exception as e:
        print(f"Error processing files: {e}")

def prepare_folders(root, targetdir):
    try:
        target_folders = os.listdir(root)
        root_folders = os.listdir(targetdir)
        target_folders.sort(reverse=False)
        root_folders.sort(reverse=False)
        if target_folders != root_folders:
            folders_to_create = [x for x in target_folders if x not in root_folders]
            for fldr in folders_to_create:
                path = os.path.join(targetdir, fldr)
                os.mkdir(path)
        return True, None
    except Exception as e:
        return False, e

if __name__ == "__main__":
    root = "D:\\10. SRH_Academia\\1. All_Notes\\2. Semester 2\\3. Artificial Intelligence\\Project\\DATA\\genres_original\\"
    targetdir = "D:\\10. SRH_Academia\\1. All_Notes\\2. Semester 2\\3. Artificial Intelligence\\Project\\DATA\\MFCC\\"
    folders_readiness, error_details = prepare_folders(root, targetdir)
    if folders_readiness:
        multi_proc_inititation(root, targetdir)
    else:
        print(f"Error preparing folders: {error_details}")



