import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
import json
import paho.mqtt.client as mqtt
import time
import yaml
import ast
import os
import sys
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf  
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Conv1D, Dropout, Reshape, MaxPooling1D, Flatten
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical

import random

# SEED_Value = 2022
# tf.random.set_seed(SEED_Value)
# os.environ['PYTHONHASHSEED']=str(SEED_Value)
# random.seed(SEED_Value)
# np.random.seed(SEED_Value)

"""
The server need to be executed first so that it can take care of no of connected 
devices based on connect status
"""
borker_address = "192.168.0.174" #"192.168.0.174"
borker_port = 1883
keep_alive = 8000
topic_train = "train"
topic_aggregate = "aggregate"
topic_initilize = "initlize"
client_name = "server"
client_queue = []
clients_per_iteration = []
wait_time = 4 #It has nothing to do with struggles
no_connected_device = 0
no_iteration = 200
iter = 0 
minimimum_clients = 1



#Utility function to convert model(h5) into string
def encode_file(file_name):
    with open(file_name,'rb') as file:
        encoded_string = base64.b64encode(file.read())
    return encoded_string

def waitFunction():
    print(f"Waiting for clients to fininsh their execution")
    while True:
        time.sleep(wait_time)
        if len(client_queue) >= minimimum_clients:
            break  

def getTestDataset(name):
    if name == 'mnist':

        test_data_dir = 'test'
        image_files = os.listdir(test_data_dir)
        test_images = []
        test_labels = []
        for image_file in image_files:
            if '.png' not in image_file:
                continue
            else :
                test_labels.append(int(image_file.split('_')[0]))
                test_image = cv2.imread(os.path.join(test_data_dir, image_file), cv2.IMREAD_GRAYSCALE)
                test_images.append(test_image)

        test_images = np.array(test_images, dtype='float32')
        test_labels = np.array(test_labels)
        test_images = test_images / 255
        test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
        test_labels = to_categorical(test_labels, num_classes=10)
        # (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
        # test_images = test_images.astype('float32')
        # test_images /=255
        # test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
        # test_labels = to_categorical(test_labels)
        return test_images, test_labels
    
    elif name == 'radar':
        database = sio.loadmat('../IEEE for federated learning/data_base_all_sequences_random.mat')
        X = database['Data_test_2']
        y = database['label_test_2']
        y = to_categorical(y, num_classes=8)
        return X, y
    else:
        return np.array([]), np.array([])


def saveLearntMetrices(modelName):
     
 # print("Inside save paramertes")
    model = load_model(modelName)
    X_test, y_test = getTestDataset('mnist')
    score = model.evaluate(X_test, y_test, verbose=1)
    print("Agreegated model with all data loss : {} and accuracy : {}".format(score[0], score[1]))

    with open('Models/globalMetrics.txt','r+') as f:
        trainMetrics = json.load(f)
        trainMetrics['accuracy'].append(score[1])
        trainMetrics['loss'].append(score[0])
        f.seek(0) 
        f.truncate()
        f.write(json.dumps(trainMetrics))

def async_w_fedAvg():

    global client_queue
    waitFunction()
    models = list()
    staleness_weights = list()
    for param in client_queue:
        payload_dict = eval(param)
        with open("Models/temp_model.h5","wb") as file:
            file.write(base64.b64decode(payload_dict['model_file']))
        model = load_model("Models/temp_model.h5")
        print(f"Model iter is  {payload_dict['iteration']} and current iter is {iter}")
        staleness_weights.append(iter - payload_dict['iteration'])
        models.append(model)
    client_queue = []
    clients_per_iteration.append(len(models))
    print(clients_per_iteration)
    weights = [model.get_weights() for model in models]
    staleness_weights = 1/(np.array(staleness_weights) + 1)
    new_weights = list()
    for weights_list_tuple in zip(*weights):
        new_weights.append(np.array([np.average(weights_, axis=0, weights=staleness_weights) for weights_ in zip(*weights_list_tuple)]))

    new_model = models[0]
    new_model.set_weights(new_weights)
    new_model.save("Models/model.h5")

    print("Averaged over all models - optimised model saved!")
    saveLearntMetrices("Models/model.h5")

def print_heat_map(array):
    fig = plt.figure(figsize=(6, 4))

    # Create a heatmap using matplotlib
    plt.imshow(array,  cmap='hot')
    plt.colorbar()  # Add a colorbar to indicate the scale
    plt.show()  # Display the heatmap

def validateWeights():
    global client_queue
    waitFunction()
    models = list()
    for param in client_queue:
        payload_dict = eval(param)
        with open("Models/temp_model.h5","wb") as file:
            file.write(base64.b64decode(payload_dict['model_file']))
        model = load_model("Models/temp_model.h5")
        print(f"Model iter is  {payload_dict['iteration']} and current iter is {iter}")
        #Discarding strugglers
        # dealy = iter - payload_dict['iteration']
        # if  dealy == 1 :
        #     models.append(model)
        models.append(model)
    #The server need to stay if all current model is stealth
    if len(models) < minimimum_clients :
        return; 
    client_queue = []
    clients_per_iteration.append(len(models))
    print(clients_per_iteration)
    weights = [model.get_weights() for model in models]

    print("------------------Showing the weights matrix------------------")
    x_train, y_train = getTestDataset('mnist')
    model_combine = [models[0]]
    for aModel in model_combine:
        # for layer in aModel.layers:
        #     print(layer.output)
        aModel_weights = aModel.get_weights()
        # weights_list_tuple = np.array(weights_list_tuple)
        # temp_ = weights_list_tuple[-2]
        # temp_ = temp_.transpose()
        # print(aModel.layers[-2].output)
        # print(aModel_weights[-2])

        outputs = [layer.output for layer in aModel.layers]
        visualization_model = tf.keras.models.Model(inputs=aModel.input, outputs=outputs)
        layer_outputs = visualization_model.predict(x_train[:1])

        # for layer_output in layer_outputs:
            # print(layer_output)
        x = np.array(layer_outputs[-2])
        w = np.array(aModel_weights[-2])
        # print(x)
        print(f'shape of x is  {x.shape}')
        # print(w)
        print(f'shape is w is  {w.shape}')
        out = np.matmul(x, w)
        print(f'shape of out is {out.shape}')

        print_heat_map(x)
        print_heat_map(w)
        print_heat_map(out)
        print(f"Output of the data is {y_train[:1]}")
        input()
    
    input("wait of a key press")


    new_weights = list()
    for weights_list_tuple in zip(*weights):
        new_weights.append(
            np.array([np.array(weights_).mean(axis=0) for weights_ in zip(*weights_list_tuple)]))

    new_model = models[0]
    new_model.set_weights(new_weights)
    new_model.save("Models/model.h5")

    print("Global weights are ")
    # print(new_weights[-2:])
    
    print("Averaged over all models - optimised model saved!")
    saveLearntMetrices("Models/model.h5")




# This fucntion aggregates all models parmeters and create new optimized model 
def fedAvg():

    global client_queue
    waitFunction()
    models = list()
    for param in client_queue:
        payload_dict = eval(param)
        with open("Models/temp_model.h5","wb") as file:
            file.write(base64.b64decode(payload_dict['model_file']))
        model = load_model("Models/temp_model.h5")
        print(f"Model iter is  {payload_dict['iteration']} and current iter is {iter}")
        #Discarding strugglers
        # dealy = iter - payload_dict['iteration']
        # if  dealy == 1 :
        #     models.append(model)
        models.append(model)
    #The server need to stay if all current model is stealth
    if len(models) < minimimum_clients :
        return; 
    client_queue = []
    clients_per_iteration.append(len(models))
    print(clients_per_iteration)
    weights = [model.get_weights() for model in models]

    new_weights = list()
    for weights_list_tuple in zip(*weights):
        new_weights.append(
            np.array([np.array(weights_).mean(axis=0) for weights_ in zip(*weights_list_tuple)]))

    new_model = models[0]
    new_model.set_weights(new_weights)
    new_model.save("Models/model.h5")

    print("Averaged over all models - optimised model saved!")
    saveLearntMetrices("Models/model.h5")

def createInitialModel():

    print("creating new model")
    model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), activation='relu', input_shape=(28, 28,1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(120, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    

    model.save('Models/model.h5')
    model.save('Models/initModel.h5')
    model.summary()

def createModelForComparision():
    print("Taking model from initModel not creating new one")
    model = load_model("Models/initModel.h5")
    model.save('Models/model.h5')
    model.summary()
        

def initlizeGlobalMetrics():
    metric = {'accuracy' : [], 'loss' : []}
    
    with open('Models/globalMetrics.txt', "w") as f:
        f.write(json.dumps(metric))


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker!")
        client.subscribe(topic_aggregate, 2)
        client.subscribe("connected", 2)
        client.subscribe("disconnected", 2)   
    else:
        print("Failed to connect, return code ", rc)

def on_message(client, userdata, msg):
    global no_connected_device
    if msg.topic == topic_aggregate :
        print("Training completed Client ", client)
        client_queue.append(msg.payload)
    if msg.topic == "connected" :
        no_connected_device = no_connected_device + 1
        print("Connected, devices are ", no_connected_device)
    if msg.topic == "disconnected":
        no_connected_device = no_connected_device - 1
        print("Disconnected, devices are ", no_connected_device)

def on_subscribe(client, userdata, mid, granted_qos):
    print("Subscribe to aggregate", client)

def main():

    mqttc = mqtt.Client(client_name)   
    # Assign event callbacks
    mqttc.on_connect = on_connect
    mqttc.on_message = on_message
    mqttc.on_subscribe = on_subscribe
    mqttc.username_pw_set("server","cloud")
    mqttc.connect(borker_address, borker_port, keep_alive)
    
    initlizeGlobalMetrics()
    saveLearntMetrices("Models/model.h5")
    mqttc.loop_start()
    input("Press any key to start training")
    print("Total no of conntected devices are ", no_connected_device)
    mqttc.publish(topic_initilize, "I am initlizing")
    global iter
    while True:
        try :
            send_message = {'model_file' : encode_file("Models/model.h5"), 
                            'epochs' : 32,
                            'conti_batch' : 60,
                            'iteration' : iter}
            if iter < no_iteration : 
                mqttc.publish(topic_train, payload = str(send_message))
                print("Current iteration : ", iter)
                iter = iter + 1
                fedAvg()
                # async_w_fedAvg()  
                # validateWeights()
            else:
                break
        except KeyboardInterrupt:
            print("Keyboard intruupt")
            break
        finally:    
            with open("client_q", "w") as fp:
                json.dump(clients_per_iteration, fp)
            print('Clients per iteration saved')
    mqttc.loop_stop()
    mqttc.disconnect()


if __name__ == '__main__':

    if sys.argv[1] == 'newModel':
        createInitialModel()
        # Use asynch_w_fedAvg as a aggregator
    elif sys.argv[1] == 'oldModel':
        createModelForComparision()
    else:
        print('Invalid model argument')
        sys.exit(0)
    main()