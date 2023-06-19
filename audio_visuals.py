import librosa
import matplotlib.pyplot as plt
import numpy as np
import os

root = "D:\\10. SRH_Academia\\1. All_Notes\\2. Semester 2\\3. Artificial Intelligence\\Project\\DATA\\Data\\genres_original\\"
targetdir = "D:\\10. SRH_Academia\\1. All_Notes\\2. Semester 2\\3. Artificial Intelligence\\Project\\DATA\\Data\\images_original\\"
targpath = [x[0] for x in os.walk(targetdir)]
del targpath[0]
i = -1
for path, subdirs, files in os.walk(root):
    for name in files:
        filename = os.path.join(path, name)
        figname = targpath[i] + "\\" + name[:len(name)-4]
        plt.figure()
        plt.subplot(1, 1, 1)
        y, sr = librosa.load(filename)
        librosa.display.waveshow(y, sr=sr, axis='time', color='cyan',)
        plt.title("Waveform")
        plt.savefig(figname + "_Wav.JPEG")
        plt.close()

        # Visualizing MFCC

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax = 8000)
        fig, ax = plt.subplots(nrows=1, sharex = True)
        img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                                       x_axis='time', y_axis='mel', fmax=8000)
        fig.colorbar(img)
        ax.set(title='Mel spectrogram')
        ax.label_outer()
        plt.savefig(figname + "_Mel_Spec.JPEG")
        plt.close()

        fig, ax = plt.subplots(nrows=1, sharex = True)
        img = librosa.display.specshow(mfccs, x_axis='time')
        fig.colorbar(img)
        ax.set(title='MFCC')
        ax.label_outer()
        plt.savefig(figname + "_MFCC.JPEG")
        plt.close()

        # Spectrogram

        fig, ax = plt.subplots()
        D_highres = librosa.stft(y, hop_length=256, n_fft=4096)
        S_db_hr = librosa.amplitude_to_db(np.abs(D_highres), ref=np.max)
        img = librosa.display.specshow(S_db_hr, hop_length=256, x_axis='time', y_axis='log',
                                       ax=ax)
        ax.set(title='spectrogram')
        fig.colorbar(img, ax=ax, format="%+2.f dB")
        plt.savefig(figname + "_Spec.JPEG")
        plt.close()
    i +=1


