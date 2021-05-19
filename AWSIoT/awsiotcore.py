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

print("Publishing Message from RPI")
myMQTTClient.publish(
   topic="home/helloworld",
   QoS=1,
   payload="{Message:Message By RPI'}"
)

