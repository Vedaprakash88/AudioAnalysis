# importing necessary packages

import imghdr
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
import math


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Removing images with incompatible extensions
datadir = "D:\\10. SRH_Academia\\1. All_Notes\\2. Semester 2\\3. Artificial Intelligence\\Project\\DATA\\Mel_spec\\"
image_exts = ['jpeg', 'jpg', 'bmp', 'png']

''' Getting the list of classes of training dataset. These classes are the folder names i.e., all the images inside 
'blues' folder belong to 'blues' class and so on.

The sequential order of these class labels is very important - the classes are encoded numerically based on the sequence
as stored in the list for example. i.e., Used to control the order of the classes (otherwise alphanumerical order is used).
(must match names of subdirectories).
class names =          ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
encoded class names=   [0,        1,           2,         3,       4,         5,      6,      7,      8,       9]

class names =         ['rock', 'reggae', 'pop', 'metal', 'jazz', 'hiphop', 'disco', 'country', 'classical', 'blues']
encoded class names=   [0,        1,           2,         3,       4,         5,      6,      7,      8,       9]

if nothing is passed, the folder / class names are encoded in alphanumerically ascending order.

THIS ONLY VALID IF THE "LABELS=" KEYWORD ARGUMENT IS SET TO "INFERRED" - labels are generated from the directory structure
and should match with the class names we pass.

'''

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
# data pipeline using director ("Found 999 files belonging to 10 classes")
# data object contains [image data , labels]
# image data = pixel values in 256x256x3 (width x height x depth (2 for black and white, 3 for colour))

''' the following is for me to understand how to use batches and not really useful
 for this project. 
 
 HOWEVER, we cannot access the data inside a tf dataset simply, so we will need to use as_numpy_iterator
 and create a batch and then access data as numpy array'''

data_iterator = data.as_numpy_iterator()  # coverts the pipeline to a numpy iterator
batch = data_iterator.next()  # grabbing data in batches of 32 (default) we need to run this in loop if we need to work
# batch-wise. but keras built-in batching system will take care, and we do not need to loop it :-)

#############################################################
# print(batch[0].shape, batch[1])
# this shows that the batch size is 32 (default), image reshaped to 256x256 size,
# is a 3D (colour pic) and the list of image classes of images selected in the batch
fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])
plt.show()

''' batching complete '''

########################################################################################################################
# Pre-processing images is necessary to decrease training complexity (Generalization) and improve prediction results.
# This is to reduce the image pixel values to between 0 and 1 instead of 0 and 255 (or max pixel number in the image)
divisor = batch[0].max()
data = data.map(lambda x, y: (x / divisor, y))  # x here is image pixel size and y is labeled
########################################################################################################################

########################################################################################################################
# Splitting the data into train, validate and test data

length_of_dataset = len(data)
train_length = int(math.ceil(len(data)*0.7))
validation_length = int(math.floor(len(data)*0.2))
test_length = int(math.floor(len(data)*0.1))

train_data = data.take(train_length)
val_data = data.skip(train_length).take(validation_length)
test_data = data.skip(train_length + validation_length).take(test_length)

########################################################################################################################

########################################################################################################################
# CNN
model = Sequential()

model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile('adam', loss=tf.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
print(model.summary())




