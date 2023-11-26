import pandas as pd
import numpy as np
import base64
import os
import cv2
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from random import randrange
import paho.mqtt.client as mqtt
import time
import sys
import json
import ast
import random

borker_address = "192.168.0.174"
borker_port = 1883
print("Connecting broker is ", borker_address, borker_port)
keep_alive = 8000
topic_train = "train"
topic_aggregate = "aggregate"
topic_initilize = "initlize"
client_name = os.getenv('name')
print(f'Client name is - {client_name}')
last_param = ''
sleep_time = 20

#This function is used fro dataset generation
def InitilizeClients():
    metric = {'accuracy' : [], 'loss' : []}
    index = {'index' : 0}

    with open('Models/indexFile.txt', "w") as f:
        f.write(json.dumps(index))
    with open('Models/metrics.txt', "w") as f:
        f.write(json.dumps(metric))
    with open('Models/localMetrics.txt', "w") as f:
        f.write(json.dumps(metric))

    print("Devices initilization done")

def saveLearntMetrice(file_name,score):

    with open(file_name,'r+') as f:
        trainMetrics = json.load(f)
        trainMetrics['accuracy'].append(score[1])
        trainMetrics['loss'].append(score[0])
        f.seek(0) 
        f.truncate()
        f.write(json.dumps(trainMetrics))

def getData(model_name, continous_flag, continousTrainingBatchSize):

    if model_name == 'mnist':
        data_dir = 'data/'
        image_files = os.listdir(data_dir)
        images = []
        labels = []
        for image_file in image_files:
            if '.png' not in image_file:
                continue
            else :
                labels.append(int(image_file.split('_')[0]))
                image = cv2.imread(os.path.join(data_dir, image_file), cv2.IMREAD_GRAYSCALE)
                images.append(image)

        images = np.array(images, dtype='float32')
        labels = np.array(labels)
        images = images / 255
        images = images.reshape((images.shape[0], 28, 28, 1))
        labels = to_categorical(labels, num_classes=10)
        # print(images.shape, labels.shape)

        currentIndex = 0
        with open('Models/indexFile.txt', "r+") as f:
            fileIndex = json.load(f)
            currentIndex = fileIndex['index']

        print("Current Index is ", currentIndex)

        
        totalRowCount = images.shape[0]
        nextIndex = currentIndex + continousTrainingBatchSize if currentIndex + continousTrainingBatchSize < totalRowCount else totalRowCount
        images = images[currentIndex:nextIndex]
        labels = labels[currentIndex:nextIndex]
        
        #Updating Index
        if nextIndex == totalRowCount:
            nextIndex = 0
        with open('Models/indexFile.txt', "w") as f: 
            index = {'index' : nextIndex}
            f.write(json.dumps(index))

        #Selecting random data
        # index = np.random.choice(images.shape[0], continousTrainingBatchSize, replace=False)  
        # images = images[index]
        # labels = labels[index]


        return images, labels
    else:
        print("Dataset argument missing")
        return 


"Training module"
def train(payload):
    # print(f"Starting training on {client_name}")
    #Creating random struggler 25% 
    # if random.randint(1, 4) % 4 == 0:
    #     time.sleep(sleep_time)
    #Creating fixed strugglers eg. client 3 and client 5
    if client_name == 'client3' or client_name == 'client5':
        time.sleep(sleep_time)
    print(f'Struggler time is {sleep_time}')
    payload = last_param
    payload_dict = eval(payload)
    epochs = payload_dict['epochs']
    try :

        with open("Models/current_model.h5","wb") as file:
            file.write(base64.b64decode(payload_dict['model_file']))
        model = load_model("Models/current_model.h5")

        # X, y = getData('radar', True, payload_dict['conti_batch'])
        X, y = getData('mnist', False, payload_dict['conti_batch'])
        print("Shape of the data is ", X.shape, y.shape)
    

        #Printing aggregated global model metrics
        score = model.evaluate(X, y, verbose=0)
        print("Global model loss : {} Global model accuracy : {}".format(score[0], score[1]))
        saveLearntMetrice('Models/metrics.txt', score)

        model.fit(X, y, batch_size=32, epochs=epochs, shuffle=True, verbose=0)
    except Exception as e:
        print(e)
        print("Error in training in current iteration")
        return str(payload_dict)  
           
    #Printing loss and accuracy after training 
    score = model.evaluate(X, y, verbose=0)
    print("Local model loss : {} Local model accuracy : {}".format(score[0], score[1]))
    saveLearntMetrice('Models/localMetrics.txt', score)

    # #Save current model 
    model.save('Models/model.h5')
    with open('Models/model.h5','rb') as file:
        encoded_string = base64.b64encode(file.read())
    payload_dict['model_file'] = encoded_string
    print(f"Local training completed at {client_name}")
    return str(payload_dict)

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker!")
        client.publish("connected", "I am connected", 2)
        client.subscribe(topic_train, 2)
        client.subscribe(topic_initilize, 2)
    else:
        print("Failed to connect, return code ", rc)

def on_message(client, userdata, msg):
        global last_param
        if msg.topic == topic_train:
            last_param = msg.payload
            param = train(msg.payload)
            client.publish(topic_aggregate, param)
        elif msg.topic == topic_initilize:
            InitilizeClients()
        else:
            print('Unknown topic', msg.topic)



def on_subscribe(client, userdata, mid, granted_qos):
    print("Subscriptioin done")

mqttc = mqtt.Client(client_name)  
mqttc.will_set("disconnected", "LOST_CONNECTION", 0, False)
mqttc.on_connect = on_connect
mqttc.on_message = on_message
mqttc.on_subscribe = on_subscribe

mqttc.username_pw_set("client","smart")
mqttc.connect(borker_address, borker_port, keep_alive)
mqttc.loop_forever()
