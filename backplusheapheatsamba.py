

import serial 
import numpy as np 

import matplotlib.pyplot as plt
import seaborn as sns

PORT_H = '/dev/ttyUSB1' 
PORT_B = '/dev/ttyUSB0' 

BaudRate = 9600 

ARD_H = serial.Serial(PORT_H,BaudRate) 
ARD_B = serial.Serial(PORT_B,BaudRate) 

array=[]
buf=[]

cnt = 33

#Arduino for heap
def Ardread_H(cnt): # return list [Ard1,Ard2] 

    if ARD_H.readable(): 
        LINE = ARD_H.readline()
        data = LINE.decode('utf-8')
        data = data.strip('\n')
        data = data.strip('\r')
        splitData = data.split(',')

        for i in range(0,16) :
            splitData[i] = int(splitData[i])

        splitData = np.reshape(splitData,(4,4))

        ax = sns.heatmap(splitData, cmap='Blues', cbar=False , vmin = 0, vmax = 1000)
        ax.tick_params(left=False, bottom=False)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
     
        fname = 'heapright' + str(cnt) +'.png'
        plt.savefig("/home/pi/shareSamba/heapright/"+fname, dpi=200)
        
    else : 
        print("읽기 실패 from _Ardread_") 
        
#Arduino for back
def Ardread_B(cnt): # return list [Ard1,Ard2] 

    if ARD_B.readable(): 
        LINE = ARD_B.readline()
        data = LINE.decode('utf-8')
        data = data.strip('\n')
        data = data.strip('\r')
        splitData = data.split(',')

        for i in range(0,9) :
            splitData[i] = int(splitData[i])

        splitData = np.reshape(splitData,(3,3))

        ax = sns.heatmap(splitData, cmap='Blues', cbar=False , vmin = 0, vmax = 1000)
        ax.tick_params(left=False, bottom=False)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
     
        fname = 'backright' + str(cnt) +'.png'
        plt.savefig("/home/pi/shareSamba/backright/"+fname, dpi=200)
        
    else : 
        print("읽기 실패 from _Ardread_")

while (True): 
    Ardread_H(cnt)
    Ardread_B(cnt)
    cnt += 1
