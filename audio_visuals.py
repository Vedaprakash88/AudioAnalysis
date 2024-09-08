import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from PIL import Image
from io import BytesIO


root = "D:\\10. SRH_Academia\\1. All_Notes\\2. Semester 2\\3. Artificial Intelligence\\Project\\DATA\\genres_original"
targetdir = "D:\\10. SRH_Academia\\1. All_Notes\\2. Semester 2\\3. Artificial Intelligence\\Project\\DATA\\Temp\\"

target_folders = os.listdir(root)
root_folders = os.listdir(targetdir)
target_folders.sort(reverse=False)
root_folders.sort(reverse=False)
img_dict = {}

if target_folders != root_folders:
    folders_to_create = [x for x in target_folders if x not in root_folders]
    for fldr in folders_to_create:
        path = os.path.join(targetdir, fldr)
        os.mkdir(path)
total_files = sum([len(files) for r, d, files in os.walk(root)])
batch = 100
loop = 1

def fig_to_image(fig):
    buf = BytesIO()
    fig.savefig(buf, format='JPEG', bbox_inches='tight', pad_inches=-0.1)
    buf.seek(0)
    img = Image.open(buf)
    # img = img.convert('RGB') 
    return img


with tqdm(total=total_files, desc="Processing audio files") as pbar:
    for pathi, subdirs, files in os.walk(root):
        for namei in files:
            filename = os.path.join(pathi, namei)
            yi, sri = librosa.load(filename)
            img_dict.update({namei: {"y": yi,
                                    "sr": sri,
                                    "path": pathi}
                            })

    if len(img_dict) > 0:
        for key in img_dict.keys():
            y = img_dict.get(key, {}).get("y")
            sr = img_dict.get(key, {}).get("sr")
            path = img_dict.get(key, {}).get("path")
            name = key



            target_folder = os.path.join(targetdir, os.path.basename(os.path.normpath(path)))
            fig_name = (os.path.join(target_folder, name[:len(name) - 4]))


            # waveform

            fig, ax = plt.subplots()
            librosa.display.waveshow(y, sr=sr, ax=ax)
            ax.set_axis_off()
            img = fig_to_image(fig)
            img.save(fig_name + "_Wav.JPEG", "JPEG")
            plt.close(fig)

            # Visualizing MFCC

            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            fig, ax = plt.subplots()
            librosa.display.specshow(mfccs, x_axis='time', ax=ax)
            ax.set_axis_off()
            img = fig_to_image(fig)
            img.save(fig_name + "_MFCC.JPEG", "JPEG")
            plt.close(fig)

            # Mel Spectrogram
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
            fig, ax = plt.subplots()
            librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                                        x_axis='time', y_axis='mel', fmax=8000, ax=ax)
            ax.set_axis_off()
            img = fig_to_image(fig)
            img.save(fig_name + "_Mel_Spec.JPEG", "JPEG")
            plt.close(fig)

            # Spectrogram
            D_highres = librosa.stft(y, hop_length=256, n_fft=4096)
            S_db_hr = librosa.amplitude_to_db(np.abs(D_highres), ref=np.max)
            fig, ax = plt.subplots()
            librosa.display.specshow(S_db_hr, hop_length=256, x_axis='time', y_axis='log', ax=ax)
            ax.set_axis_off()
            img = fig_to_image(fig)
            img.save(fig_name + "_Spec.JPEG", "JPEG")
            plt.close(fig)

            pbar.update(1)


