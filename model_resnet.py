import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

#image header
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True

#keras header
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping #테스트셋 오차가 줄지 않으면 학습을 멈추게 하는 함수 호출
from keras.callbacks import ModelCheckpoint #모델을 저장하기 위해 호출
from keras.callbacks import EarlyStopping

#resnet header
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense


RESNET50_POOLING_AVERAGE = 'avg'
DENSE_LAYER_ACTIVATION = 'softmax'
OBJECTIVE_FUNCTION = 'categorical_crossentropy'

class_names = ['correctPosition','father_Legs','leftLean_posture','mother_leftLegs','mother_rightLegs','rightLean_posture','twist_leftLegs','twist_rightLegs']
TRAINING_DIR = "C:/buster/dataset2/train"
VALIDATION_DIR = "C:/buster/dataset2/test"

batch_size = 25

# Image Data Generator with Augmentation
training_datagen = ImageDataGenerator(rescale=1./255)   # 0~1 사이의 값으로 만들어줌
validation_datagen = ImageDataGenerator(rescale=1./255)

# Reading images from directory and pass them to the model
train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,                       # 내부 폴더에 있는 이미지도 각각 읽어옴
    batch_size=batch_size,              # 한번에 몇개 읽을 지
    target_size=(224, 224),             # input 이미지 사이즈
    class_mode='categorical',           # 폴더의 리스트를 쭉 읽어서 개수만큼 클래스 생성
    shuffle=True                        # 데이터의 순서를 랜덤하게 읽어오는 것
)

# test set
validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    batch_size=batch_size,
    target_size=(224, 224),
    class_mode='categorical'
)

# Plotting the augmented images
# img, label = next(train_generator)
# plt.figure(figsize=(20, 20))
# a=np.array([1,2,3,4,5,6,7,8])


#모델링
model = Sequential()

model.add(ResNet50(include_top = False, pooling = RESNET50_POOLING_AVERAGE, weights = 'imagenet'))
model.add(Dense(8, activation = DENSE_LAYER_ACTIVATION)) # add softmax

# Say not to train first layer (ResNet) model as it is already trained
model.layers[0].trainable = True

model.summary()

with tf.device('/gpu:0'):
  model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
              metrics=['accuracy'])

# 모델 저장 폴더 만들기
  MODEL_DIR = './model/'
  if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

  modelpath="./model/{epoch:02d}-{val_loss:.4f}.hdf5" # 에포크 횟수와 테스트셋 오차 값을 이용하여 파일명 생성, 확장자 : hd5

# 모델 업데이트 및 저장
  checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)

# 학습 자동 중단 설정
  early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5)

  # Training
  model.fit(train_generator,epochs=20, validation_data=validation_generator, verbose=1,callbacks=[early_stopping_callback,checkpointer] )

  y = model.predict("C:/buster/dataset2/train/correctPosition/correct6.png")
  print(y)


#0616 : 완성된 모델을 이용하여 데이터셋을 잘 분류하는지 확인하기
#0620 : 실시간으로 들어오는 데이터 분류
