import boto3 
from botocore.exceptions import NoCredentialsError

ACCESS_KEY = 'AKIAX4F3GCZSM6TRSRXS'
SECRET_KEY = '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'

def upload_to_aws(local_file, bucket, s3_file):
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)

    try:
        print(s3.upload_file(local_file, bucket, s3_file))
        return True
    except FileNotFoundError:
        return False
    except NoCredentialsError:
        return False


import serial 
import numpy as np 

import matplotlib.pyplot as plt
import seaborn as sns

PORT = '/dev/ttyUSB0' 
BaudRate = 9600 

ARD = serial.Serial(PORT,BaudRate) 

array=[]
buf=[]

cnt = 0

def Ardread(cnt): # return list [Ard1,Ard2] 

    if ARD.readable():
        LINE = ARD.readline()
        data = LINE.decode('utf-8')
        data = data.strip('\n')
        data = data.strip('\r')
        splitData = data.split(',')
        print(splitData)
 
        for i in range(0,16) :
            splitData[i] = int(splitData[i])

        splitData = np.reshape(splitData,(4,4))

        ax = sns.heatmap(splitData, cmap='Blues', cbar=False , vmin = 0, vmax = 1000)
        ax.tick_params(left=False, bottom=False)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

        fname = 'savefig_200dpi' + str(cnt) +'.png'
        plt.savefig(fname, dpi=200)
        uploaded = upload_to_aws(fname, 'bucektmin', fname)


 
    else :
        print("Fail from _Ardread_") 
        
while (True): 
    Ardread(cnt)
    cnt+=1
