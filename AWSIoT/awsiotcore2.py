import time
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient

def helloworld(self, params, packet):
 print ('Recieved Massage:')
 print ('Topic: '+packet.topic)
 print ("Payload: ", (packet.payload))

myMQTTClient = AWSIoTMQTTClient("MinClientID") #random key, if another connection using the same key is opened the previous one is auto closed by AWS IOT
myMQTTClient.configureEndpoint("a2up5njeqojvfd-ats.iot.us-west-2.amazonaws.com", 8883)

myMQTTClient.configureCredentials("/home/pi/AWSIoT/root-ca.pem", "/home/pi/AWSIoT/private.pem.key", "/home/pi/AWSIoT/certificate.pem.crt")

myMQTTClient.configureOfflinePublishQueueing(-1) # Infinite offline Publish queueing
myMQTTClient.configureDrainingFrequency(2) # Draining: 2 Hz
myMQTTClient.configureConnectDisconnectTimeout(10) # 10 sec
myMQTTClient.configureMQTTOperationTimeout(5) # 5 sec
print ('Initiating Realtime Data Transfer From Raspberry Pi...')
myMQTTClient.connect()
myMQTTClient.subscribe("home/helloworld", 1, helloworld)

while True:
	time.sleep(5)
