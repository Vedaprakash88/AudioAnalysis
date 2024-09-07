import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.models import load_model
import numpy as np
import cv2
import imghdr

img_path = "D:\\10. SRH_Academia\\1. All_Notes\\2. Semester 2\\3. Artificial Intelligence\\Project\\DATA\\Mel_spec\\jazz\\jazz.00026_Mel_Spec.JPEG"
Mel_model_path = "D:\\10. SRH_Academia\\1. All_Notes\\2. Semester 2\\3. Artificial Intelligence\\Project\\DATA\\models\\audio_classifier_Mel_spec_VCv1.h5"
# MFCC_model_path = "D:\\10. SRH_Academia\\1. All_Notes\\2. Semester 2\\3. Artificial Intelligence\\Project\\DATA\\models\\audio_classifier_MFCC_VCv1.h5"
new_model = load_model(Mel_model_path,  compile=False)

img = cv2.imread(img_path)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

re_img = tf.image.resize(img, (256, 256))
print(re_img.shape)
plt.imshow(re_img.numpy().astype(int))
plt.show()

norm_img = re_img/255
bt_img = np.expand_dims(re_img, 0)
print(bt_img.shape)
yhat = new_model.predict(bt_img)
print(yhat)
