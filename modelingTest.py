from numpy.lib import type_check
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
import matplotlib.pyplot as plt

# Set your dataset directory
# Directory Structure:
# -- train-set
# ------------/on_mask
# ------------/off_mask
# --- test-set
# ------------/on_mask
# ------------/off_mask

class_names = ['correctPosition','father_Legs','leftLean_posture','mother_leftLegs','mother_rightLegs','rightLean_posture','twist_leftLegs','twist_rightLegs']
TRAINING_DIR = "C:/buster/dataset2/test"
VALIDATION_DIR = "C:/buster/dataset2/train"

batch_size = 50

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

for i in range(50):
    plt.subplot(10,5, i+1)
    plt.imshow(img[i])
    plt.title(class_names[int(sum(label[i]*a))-1])
    plt.axis('off')

plt.show()

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

model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])
   
# Training
history = model.fit_generator(train_generator,epochs=50, validation_data=validation_generator, verbose=1)

print("hihihih-------------------------------------------------------ihih")                
print("\n Test Accuracy: %.4f" % (model.evaluate(validation_generator, class_names)))
# Save the trained model
model.save("saved_model.h5")
