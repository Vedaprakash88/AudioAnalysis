import imghdr
import os
import matplotlib.pyplot as plt
import tensorflow as tf  # V2.12.0
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.python.keras.metrics import SparseCategoricalAccuracy
from tensorflow.python.keras.optimizer_v2 import adam
from tensorflow.python.keras.models import load_model
import math
import easygui

datadir = easygui.diropenbox(msg="Select folder with audio-images for training", title="Audio Classification")
image_exts = ['jpeg', 'jpg', 'bmp', 'png']
class_names = os.listdir(datadir)
class_names.sort(reverse=False)

for path, subdirs, files in os.walk(datadir):
    for image in files:
        image_path = os.path.join(path, image)
        try:
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print('Image extension not allowed for image: {}'.format(image_path))
                os.remove(image_path)
        except Exception as e:
            print('Issue with image {}'.format(image_path))

# Loading the data - Creating data pipelines instead of loading whole data into RAM as the data is too large
data = tf.keras.utils.image_dataset_from_directory(
    datadir, color_mode="rgb", labels='inferred', class_names=class_names)

data_iterator = data.as_numpy_iterator()  # coverts the pipeline to a numpy iterator
batch = data_iterator.next()  # grabbing data in batches of 32 (default) we need to run this in loop if we need to work
# batch-wise. but keras built-in batching system will take care, and we do not need to loop it :-)

#############################################################
print(batch[0].shape, batch[1])
# this shows that the batch size is 32 (default), image reshaped to 256x256 size,
# is a 3D (colour pic) and the list of image classes of images selected in the batch
fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])
plt.show()

''' batching complete '''

########################################################################################################################
# Pre-processing images is necessary to decrease training complexity (Generalization) and improve prediction results.
# This is to reduce the image pixel values to between 0 and 1 instead of 0 and 255 (or max pixel number in the image)
divisor = batch[0].max()
data = data.map(lambda x, y: (x / divisor, y))  # x here is image pixel size and y is labelled
########################################################################################################################

########################################################################################################################
# Splitting the data into train, validate and test data

length_of_dataset = len(data)
train_length = int(math.ceil(len(data) * 0.7))
validation_length = int(math.floor(len(data) * 0.2))
test_length = int(math.floor(len(data) * 0.1))

train_data = data.take(train_length)
val_data = data.skip(train_length).take(validation_length)
test_data = data.skip(train_length + validation_length).take(test_length)

m_path = "D:\\10. SRH_Academia\\1. All_Notes\\2. Semester 2\\3. Artificial Intelligence\\Project\\DATA\\checkpoint" # \\saved_model.pb"
model = load_model(m_path,  compile=False)
acc = SparseCategoricalAccuracy()
for batch in test_data.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    # precision_l, recall_l, f1_l, _ = score(y, yhat)
    # precision.append(precision_l)
    # recall.append(recall_l)
    # f1.append(f1_l)
    acc.update_state(y, yhat)

# precision = sum(precision) / len(precision)
# recall = sum(recall) / len(recall)
# f1 = sum(f1) / len(f1)
#
# print('precision: {}'.format(precision))
# print('recall: {}'.format(recall))
# print('fscore: {}'.format(f1))
print('Accuracy: {}'.format(acc.result()))
