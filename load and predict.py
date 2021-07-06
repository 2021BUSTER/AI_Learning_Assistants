# 0. 사용할 패키지 불러오기
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
from numpy import argmax
from PIL import Image
#from skimage import transform

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing import image

import tensorflow as tf

class_names = ['correctPosition','father_Legs','leftLean_posture','mother_leftLegs','mother_rightLegs','rightLean_posture','twist_leftLegs','twist_rightLegs']
TEST_DIR = "C:/buster/dataset/test"

batch_size = 10

# Image Data Generator with Augmentation
test_datagen = ImageDataGenerator(rescale=1./255)

# test set
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    batch_size=batch_size,
    target_size=(224, 224),
    class_mode='categorical',
    shuffle=False
)

loaded_model = tf.keras.models.load_model('./model/50-1.0677.hdf5')

# 예측 값과 실제 값의 비교
filenames = test_generator.filenames
nb_samples = len(filenames)

Y_prediction = loaded_model.predict_generator(test_generator, steps = nb_samples)
print(Y_prediction[0])
cnt = 0
for i in Y_prediction:
    idx = tf.argmax(i)
    print(idx, class_names[idx],filenames[cnt])
    cnt += 1
print(test_generator.class_indices)

# prediction_idx = tf.argmax(Y_prediction[0])
# print(prediction_idx)
# print(filenames)