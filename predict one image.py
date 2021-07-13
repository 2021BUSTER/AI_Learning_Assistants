import numpy as np
from keras.preprocessing import image

# 0. 사용할 패키지 불러오기
import numpy as np
from numpy import argmax
#from skimage import transform
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

import tensorflow as tf
import matplotlib.pyplot as plt

class_names = ['correctPosition','father_Legs','leftLean_posture','mother_leftLegs','mother_rightLegs','rightLean_posture','twist_leftLegs','twist_rightLegs']

model = tf.keras.models.load_model('./model/148-0.5727.hdf5')

test_image = image.load_img('dataset/test/mother_rightLegs/mother_rightLegs97.png', target_size = (224,224))
test_image = image.img_to_array(test_image)
test_image = test_image / 255

test_image = np.expand_dims(test_image, axis=0)
predictions = model.predict(test_image)
print(predictions[0])
print(np.argmax(predictions[0]))
