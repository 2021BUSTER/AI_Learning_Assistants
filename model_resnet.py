import tensorflow as tf
import os

#image header
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True

#keras header
from keras import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping #테스트셋 오차가 줄지 않으면 학습을 멈추게 하는 함수 호출
from keras.callbacks import ModelCheckpoint #모델을 저장하기 위해 호출

#resnet header
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.layers import Dense

RESNET50_POOLING_AVERAGE = 'avg'
DENSE_LAYER_ACTIVATION = 'softmax'
OBJECTIVE_FUNCTION = 'categorical_crossentropy'

class_names = ['correctPosition','father_Legs','leftLean_posture','mother_leftLegs','mother_rightLegs','nobody','rightLean_posture','twist_leftLegs','twist_rightLegs']
TRAINING_DIR = "C:/buster/dataset/train"
VALIDATION_DIR = "C:/buster/dataset/validation"

batch_size = 50

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

# validation set
validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    batch_size=batch_size,
    target_size=(224, 224),
    class_mode='categorical',
    shuffle=False
)

#모델링
model = models.Sequential()
model.add(ResNet50(include_top = False, pooling = RESNET50_POOLING_AVERAGE, weights = 'imagenet',input_shape=(224,224,3), classes=9))
model.add(Dense(9, activation = DENSE_LAYER_ACTIVATION)) # add softmax

# Say not to train first layer (ResNet) model as it is already trained
model.layers[0].trainable = False

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
  early_stopping_callback = EarlyStopping(monitor='val_loss', patience=8)

  # Training
  model.fit(train_generator,
    epochs=200, 
    validation_data=validation_generator, 
    verbose=1,
    callbacks=[early_stopping_callback,checkpointer] )

# for i in range(8):
#     prediction = Y_prediction[i]
#     print("실제: {:.3f}, 예상: {:.3f}".format(class_name, prediction))

#1. resnet layer 늘리기
#2. softmax 전에 컨볼루션 넣기
#3. 다른 데이터셋으로 테스트
#//------ 6/21까지
#4. 데이터 추가
#5. 센서 추가(엉덩이 6x6, 등 4x4)