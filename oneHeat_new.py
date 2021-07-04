import serial 
import numpy as np 

import matplotlib.pyplot as plt
import seaborn as sns

PORT_B = '/dev/ttyUSB2' 
PORT_H = '/dev/ttyUSB0' 
PORT_H2 = '/dev/ttyUSB1'

BaudRate = 9600 

ARD_B = serial.Serial(PORT_B,BaudRate)
ARD_H = serial.Serial(PORT_H,BaudRate)  
ARD_H2 = serial.Serial(PORT_H2,BaudRate)

cnt = 0

data_9 = list(range(0,18))
splitData = list(range(0,30))

#Arduino for heap
def Ardread_H(cnt): # return list [Ard1,Ard2] 
    global splitData
    if ARD_H.readable() and ARD_H2.readable(): 
        LINE = ARD_H.readline()
        LINE2 = ARD_H2.readline()
        data = LINE.decode('utf-8')
        data2 = LINE2.decode('utf-8')
        data = data+','+data2
        data = data.strip('\n')
        data = data.strip('\r')
        print(data)
        splitData = data.split(',')

        for i in range(0,30) :
            splitData[i] = int(splitData[i])

        splitData = np.reshape(splitData,(5,6))
        splitData = np.flip(splitData)
    else :
        print("read fail from _Ardread_") 


#Arduino for back
def Ardread_B(cnt): # return list [Ard1,Ard2] 
    global data_9
    if ARD_B.readable(): 
        LINE = ARD_B.readline()
        data = LINE.decode('utf-8')
        data = data.strip('\n')
        data = data.strip('\r')
        Data9 = data.split(',')
        for i in range(0,9) :
            Data9[i] = int(Data9[i])

        j = 0
        for i in range(0,18) :
            if i % 6 >= 3 :
                data_9[i] = 0
            else :
                data_9[i] = Data9[j]
                j = j+1

        data_9 = np.reshape(data_9,(3,6))

    else : 
        print("read fail from _Ardread_")


        

def merger_heatmap() :
    global splitData
    global data_9
    
    heatmapArr = np.concatenate((data_9, splitData), axis=0)
    print(heatmapArr)

    ax = sns.heatmap(heatmapArr, cmap='Blues', cbar=False , vmin = 0, vmax = 1024)
    ax.tick_params(left=False, bottom=False)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
     
    fname = 'twist_rightLegs' + str(cnt) +'.png'
    plt.savefig("/home/pi/shareSamba/dataset2/twist_rightLegs/"+fname, dpi=200)
        
    data_9 = np.reshape(data_9,(18,1))
    splitData = np.reshape(splitData,(30,1))
    
while (True): 
    Ardread_H(cnt)
    Ardread_B(cnt)
    merger_heatmap()
    cnt += 1


