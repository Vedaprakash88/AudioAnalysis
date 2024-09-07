# importing necessary packages

import imghdr
import os
import matplotlib.pyplot as plt
import tensorflow as tf  # V2.12.0
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.python.keras.metrics import SparseCategoricalAccuracy
from tensorflow.python.keras.optimizer_v2 import adam
import math

datadir = "D:\\10. SRH_Academia\\1. All_Notes\\2. Semester 2\\3. Artificial Intelligence\\Project\\DATA\\Mel_spec\\"
model_path = ("D:\\10. SRH_Academia\\1. All_Notes\\2. Semester 2\\3. Artificial "
              "Intelligence\\Project\\DATA\\models\\MFCC\\Audio_classifier_MFCC_VCv1.keras")
logdir = "D:\\10. SRH_Academia\\1. All_Notes\\2. Semester 2\\3. Artificial Intelligence\\Project\\DATA\\log_dir\\MEL_SPEC\\"
chk_pt = "D:\\10. SRH_Academia\\1. All_Notes\\2. Semester 2\\3. Artificial Intelligence\\Project\\DATA\\checkpoint\\MEL_SPEC\\"

# datadir = easygui.diropenbox(msg="Select folder with audio-images for training", title="Audio Classification")
# model_path = easygui.diropenbox(msg="Select folder to save trained model", title="Audio Classification") logdir =
# easygui.diropenbox(msg="Select folder to save logs", title="Audio Classification") chk_pt = easygui.diropenbox(
# msg="Select folder to save checkpoints", title="Audio Classification") name = easygui.enterbox(msg="Enter your
# model name !!", title="save_model", default="e.g. audio_classifier_MFCC_VCv1.h5", strip=True, image=None, root=None)'

# model_save = os.path.join(model_path, name)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
print(gpus)
# Removing images with incompatible extensions
# datadir = "D:\\10. SRH_Academia\\1. All_Notes\\2. Semester 2\\3. Artificial Intelligence\\Project\\DATA\\MFCC"
# model_path = os.path.dirname(datadir)

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
    datadir, color_mode="rgb", labels='inferred', class_names=class_names, label_mode="int", batch_size=10)
# label_mode="int" for sparse categorical cross entropy loss function
# label_mode="categorical" for categorical cross entropy loss function. This will one-hot encode the labels


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

########################################################################################################################

########################################################################################################################
# CNN
model = Sequential()

model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3), padding='same'))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

''' 
The Input Layer

Every NN has exactly one of them.

Number of neurons in this layer is determined by the shape of the input data. 
Specifically, the number of features / columns in the training data (here its 256). 
Some NN configurations add one additional node for a bias term.

'''

model.add(Dense(512, activation='relu'))

'''
The Hidden layer

When designing a neural network, the number of neurons in each layer, including the Dense layer, is typically determined
by the complexity of the problem you're trying to solve and the size of the input data. The number of neurons can be 
adjusted as a hyperparameter during the model development process, and it often requires some experimentation and 
tuning to find the optimal architecture for a specific task.

Jeff Heaton - Introduction to Neural Networks for Java, Second Edition The Number of Hidden Layers

The Number of Hidden Layers

There are really two decisions that must be made regarding the hidden layers: how many hidden layers to actually have 
in the neural network and how many neurons will be in each of these layers. We will first examine how to determine the 
number of hidden layers to use with the neural network.

Problems that require two hidden layers are rarely encountered. However, neural networks with two hidden layers can 
represent functions with any kind of shape. There is currently no theoretical reason to use neural networks with any 
more than two hidden layers. In fact, for many practical problems, there is no reason to use any more than one hidden 
layer. Table 5.1 summarizes the capabilities of neural network architectures with various hidden layers.

            0 - Only capable of representing linear separable functions or decisions.

            1 - Can approximate any function that contains a continuous mapping
                from one finite space to another.

            2 - Can represent an arbitrary decision boundary to arbitrary accuracy
                with rational activation functions and can approximate any smooth
                mapping to any accuracy.

Deciding the number of hidden neuron layers is only a small part of the problem. You must also determine how many 
neurons will be in each of these hidden layers. This process is covered in the next section.

The Number of Neurons in the Hidden Layers

Deciding the number of neurons in the hidden layers is a very important part of deciding your overall neural network 
architecture. Though these layers do not directly interact with the external environment, they have a tremendous 
influence on the final output. Both the number of hidden layers and the number of neurons in each of these hidden 
layers must be carefully considered.

Using too few neurons in the hidden layers will result in something called underfitting. Underfitting occurs when 
there are too few neurons in the hidden layers to adequately detect the signals in a complicated data set.

Using too many neurons in the hidden layers can result in several problems. First, too many neurons in the hidden 
layers may result in overfitting. Overfitting occurs when the neural network has so much information processing 
capacity that the limited amount of information contained in the training set is not enough to train all of the 
neurons in the hidden layers. A second problem can occur even when the training data is sufficient. An inordinately 
large number of neurons in the hidden layers can increase the time it takes to train the network. The amount of training 
time can increase to the point that it is impossible to adequately train the neural network. Obviously, some compromise 
must be reached between too many and too few neurons in the hidden layers.

There are many rule-of-thumb methods for determining the correct number of neurons to use in the hidden layers, 
such as the following:

The number of hidden neurons should be between the size of the input layer and the size of the output layer.
The number of hidden neurons should be 2/3 the size of the input layer, plus the size of the output layer.
The number of hidden neurons should be less than twice the size of the input layer.
These three rules provide a starting point for you to consider. Ultimately, the selection of an architecture for 
your neural network will come down to trial and error. But what exactly is meant by trial and error? You do not want to 
start throwing random numbers of layers and neurons at your network. 

In order to secure the ability of the network to generalize the number of nodes has to be kept as low as possible. 
If you have a large excess of nodes, you network becomes a memory bank that can recall the training set to perfection 
(overfitting), but does not perform well on samples that was not part of the training set (underfitting)

'''
# based on my data I chose to go with 1 hidden layer with mean of input and output neurons, brought close to the,
# the model was overfitting/learning haphazardly, so i removed the hidden layer and accuracy reached to val_acc = 98%
# and test_acc = 100%
# bit value of input neurons
model.add(Dense(128, activation='relu'))

'''

The Output Layer

Like the Input layer, every NN has exactly one output layer. Determining its size (number of neurons) is based on the 
model:

Regressor: one node as output is just 1 value
classifier: depends on number of classes i.e., 2 nodes for Binary and 2+ for multi-class classification

'''
model.add(Dense(10, activation='softmax'))
opt_adam = adam.Adam(learning_rate=0.0001)
model.compile(optimizer=opt_adam, loss=tf.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
print(model.summary())

########################################################################################################################
# Configuring Callbacks
''' 

A callback in keras helps us in a proper training of the model. From the framework point of view it is an object that
we can pass to the model while using the fit method and can call it during different point of the training.

'''

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=7, mode='auto')
checkpoint = tf.keras.callbacks.ModelCheckpoint(chk_pt, monitor='val_accuracy', save_best_only=True)
callbacks = [tensorboard_callback, checkpoint]

# End of checkpoint definition #########################################################################################

########################################################################################################################

# Fitting training data with model and validation data

hist = model.fit(train_data, epochs=10, validation_data=val_data, callbacks=callbacks)
# print(hist.history.keys())

########################################################################################################################

fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()

fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()

# precision = []
# recall = []
# f1 = []
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
model.save(model_path)
