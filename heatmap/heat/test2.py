import serial 

PORT = '/dev/ttyUSB0' 
BaudRate = 9600 

ARD = serial.Serial(PORT,BaudRate) 

array=[]
buf=[]

def Ardread(): # return list [Ard1,Ard2] 
    if ARD.readable(): 
        LINE = ARD.readline()
        data = LINE.decode('utf-8')
        data = data.strip('\n')
        data = data.strip('\r')
        splitData = data.split(',')
        print(splitData)
    else : 
        print("읽기 실패 from _Ardread_") 
        
while (True): 
    Ardread()
