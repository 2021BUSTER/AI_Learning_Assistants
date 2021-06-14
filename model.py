import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib import type_check

#테스트셋 오차가 줄지 않으면 학습을 멈추게 하는 함수 호출
from keras.callbacks import EarlyStopping
#모델을 저장하기 위해 호출
from keras.callbacks import ModelCheckpoint
import os

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True

# from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50

from keras.callbacks import EarlyStopping

# physical_devices = tf.config.list_physical_devices('GPU')
# try:
#   tf.config.experimental.set_memory_growth(physical_devices[0], True)
# except:
#   # Invalid device or cannot modify virtual devices once initialized.
#   pass

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True 
session = tf.compat.v1.Session(config=config)

# Set your dataset directory
# Directory Structure:
# -- train-set
# ------------/on_mask
# ------------/off_mask
# --- test-set
# ------------/on_mask
# ------------/off_mask

class_names = ['correctPosition','father_Legs','leftLean_posture','mother_leftLegs','mother_rightLegs','rightLean_posture','twist_leftLegs','twist_rightLegs']
TRAINING_DIR = "C:/buster/dataset2/train"
VALIDATION_DIR = "C:/buster/dataset2/test"

batch_size = 25

# Image Data Generator with Augmentation
training_datagen = ImageDataGenerator(
      rescale=1./255, # 0~1 사이의 값으로 만들어줌
    #   brightness_range=(0.5, 1.3), # 50퍼센트 어둡게, 30퍼센트 밝게
      )

validation_datagen = ImageDataGenerator(rescale=1./255)

# Reading images from directory and pass them to the model
train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,  # 내부 폴더에 있는 이미지도 각각 읽어옴
    batch_size=batch_size, # 한번에 몇개 읽을 지
    target_size=(224, 224), # input 이미지 사이즈
    class_mode='categorical', # 폴더의 리스트를 쭉 읽어서 개수만큼 클래스 생성
    shuffle=True # 데이터의 순서를 랜덤하게 읽어오는 것
)

# test set
validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    batch_size=batch_size,
    target_size=(224, 224),
    class_mode='categorical'
)

# Plotting the augmented images
img, label = next(train_generator)
plt.figure(figsize=(20, 20))
a=np.array([1,2,3,4,5,6,7,8])

# for i in range(25):
#     plt.subplot(4,5, i+1)
#     plt.imshow(img[i])
#     plt.title(class_names[int(sum(label[i]*a))-1])
#     plt.axis('off')

# plt.show()

# Load pre-trained base model.
base_model = tf.keras.applications.ResNet50(input_shape=(224, 224, 3),
                                               include_top=False, weights='imagenet')

# Freeze the base model
base_model.trainable = False

# Add Custom layers
out_layer = tf.keras.layers.Conv2D(128, (1, 1), padding='SAME', activation=None)(base_model.output)
out_layer = tf.keras.layers.BatchNormalization()(out_layer)
out_layer = tf.keras.layers.ReLU()(out_layer) # 7x7x128
out_layer = tf.keras.layers.GlobalAveragePooling2D()(out_layer) # 128
out_layer = tf.keras.layers.Dense(8, activation='softmax')(out_layer)

# Make New Model
model = tf.keras.models.Model(base_model.input, out_layer)

model.summary()

with tf.device('/gpu:0'):
  model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])

# 모델 저장 폴더 만들기
  MODEL_DIR = './model/'
  if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

  modelpath="./model/{epoch:02d}-{val_loss:.4f}.hdf5" # 에포크 횟수와 테스트셋 오차 값을 이용하여 파일명 생성, 확장자 : hd5

# 모델 업데이트 및 저장
  checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)

# 학습 자동 중단 설정
#patience=100 : 오차가 좋아지지 않아도 100번 기다림
  early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5)

#모델 실행
# model.fit(X, Y, validation_split=0.2, epochs=3500, batch_size=500, verbose=0, callbacks=[early_stopping_callback,checkpointer])

# 학습 중단 설정
# early_stopping_callback = EarlyStopping(monitor='val_loss',patience=20)   


  # Training
  model.fit(train_generator,epochs=40, validation_data=validation_generator, verbose=1,callbacks=[early_stopping_callback,checkpointer] )

# print("\n Acuuracy: %.4f" % (model.evaluate(train_generator,class_names)[1]))

# Save the trained model
# model.save("saved_model.h5")
