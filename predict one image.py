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
import os
import time

import sqlite3
from sqlite3.dbapi2 import Date
import datetime

class_names = ['correctPosition','father_Legs','leftLean_posture','mother_leftLegs','mother_rightLegs','rightLean_posture','twist_leftLegs','twist_rightLegs']

model = tf.keras.models.load_model('./model/98-0.8714.hdf5')

file = '//raspberrypi/pi/realTimeImage/realTime_image.png'

while True:
    if os.path.isfile(file):
        time.sleep(3)
        print("들어감....................")
        test_image = image.load_img('//raspberrypi/pi/realTimeImage/realTime_image.png', target_size = (224,224))
        test_image = image.img_to_array(test_image)
        test_image = test_image / 255

        test_image = np.expand_dims(test_image, axis=0)
        predictions = model.predict(test_image) 
        print(predictions[0])
        idx = np.argmax(predictions[0])
        print(class_names[idx])
        print(int(idx))
        print(type(int(idx)))
        os.remove('//raspberrypi/pi/realTimeImage/realTime_image.png')
        print("제거 완료")
        
        now = datetime.datetime.now()
        conn = sqlite3.connect("C:\\Users\\ESE\\anaconda3\\envs\\buster\\flask-live-charts\\DB2.db")
        cur = conn.cursor() # 커서 열기
        idx=int(idx)
        if idx == 0: v=4
        if idx == 1: v=5
        if idx == 2: v=7
        if idx == 3: v=8
        if idx == 4: v=1
        if idx == 5: v=2
        if idx == 6: v=6
        if idx == 7: v=3
        

        cur.execute("INSERT INTO pose (posture,datetime) VALUES(?,?)",(v,time.time()*1000))
       
        conn.commit()
        conn.close()


# print (decode_predictions (predictions, top = 8) [0])
