import time
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient

def helloworld(self, params, packet):
  print ('Recieved Message from AWS IoT Core')
  print ('Topic: '+packet.topic)
  print("Payload: ",(packet.payload))

myMQTTClient = AWSIoTMQTTClient("MinClientID")
myMQTTClient.configureEndpoint("a2up5njeqojvfd-ats.iot.us-west-2.amazonaws.com",8883)

myMQTTClient.configureCredentials("/home/pi/AWSIoT/root-ca.pem","/home/pi/AWSIoT/private.pem.key","/home/pi/AWSIoT/certificate.pem.crt")

myMQTTClient.configureOfflinePublishQueueing(-1)
myMQTTClient.configureDrainingFrequency(2)
myMQTTClient.configureConnectDisconnectTimeout(10)
myMQTTClient.configureMQTTOperationTimeout(5)
print('Initiating IoT core Topic ...')
myMQTTClient.connect()
myMQTTClient.subscribe("home/helloworld",1,helloworld)

#while True:
#   time.sleep(5)

########################################################################
import serial 

PORT = '/dev/ttyUSB0' 
BaudRate = 9600 

ARD = serial.Serial(PORT,BaudRate) 

array=[]
buf=[]
newdata=0
data=0
def Ardread(): # return list [Ard1,Ard2] 
    if ARD.readable():
        LINE = ARD.readline()
        data = LINE.decode('utf-8')
        #data = data.strip('\n')
        #data = data.strip('\r')
        #splitData = data.split(',')
        #newdata = ''.join(data)
        print(data)
    else : 
        print("읽기 실패 from _Ardread_") 

while (True): 
   Ardread()
   print("Publishing Message from RPI")
   myMQTTClient.publish(
   topic="home/helloworld",
   QoS=1,
   ##payload="{Message:Message test By RPI'}"
   ##time.sleep(5)
   payload=data
)



