import scipy.io as sio
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image



def createRadarFLData(n_device):

    #Data preprocessing
    database = sio.loadmat('../IEEE for federated learning/data_base_all_sequences_random.mat')
    x_train = database['Data_train_2']
    y_train = database['label_train_2']
    #y_train_t = to_categorical(y_train)
    #x_train = (x_train.astype('float32') + 140) / 140 # DATA PREPARATION (NORMALIZATION AND SCALING OF FFT MEASUREMENTS)
    #x_train2 = x_train[iii * samples:((iii + 1) * samples - 1), :] # DATA PARTITION

    # x_test = database['Data_test_2']
    # y_test = database['label_test_2']
    #x_test = (x_test.astype('float32') + 140) / 140
    #y_test_t = to_categorical(y_test)

    indices = database['permut']
    indices = indices - 1 # 1 is subtracted to make 0 at index
    indices = indices[0] # Open indexing

    i = 0
    slot = len(indices)//n_device
    data = []
    folderId = 0
    if not os.path.exists('../data'):
        os.makedirs('../data')
    while i < len(indices) :
        if i//slot + 1 <= n_device:
            folderId = i//slot + 1
            data = []
        else:
            folderId = n_device

        for _ in range(slot):
            x = x_train[indices[i]]
            y = y_train[indices[i]]
            row = np.append(x, y)
            data.append(row)
            i = i + 1
            if(i == len(indices)):
                break
        if not os.path.exists('../data/node0' + str(folderId)):
            os.makedirs('../data/node0' + str(folderId))
        df = pd.DataFrame(data)
        df.reset_index()
        df.to_csv('../data/node0' + str(folderId) + '/data.csv')
    print("Dataset is created for %d devices" %(n_device))
  

def createMnistFLData(n_device):

    if not os.path.exists('../data'):
        os.makedirs('../data')
    folderId = 0

    #Data preprocessing
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    slot = len(train_images)//n_device

    # Save the train images.
    for i in range(len(train_images)):
        
        if i % slot == 0:
            folderId += 1
            if not os.path.exists('../data/node0' + str(folderId) + '/data_mnist'):
                os.makedirs('../data/node0' + str(folderId) + '/data_mnist')       

        image = train_images[i]
        image = image.reshape((28, 28))
        image = image.astype('uint8')
        image = Image.fromarray(image)
        image.save('../data/node0{}/data_mnist/{}_{}.png'.format(folderId,train_labels[i],i))

    # Save the test images.
    if not os.path.exists('../data/test'):
        os.makedirs('../data/test')
    for i in range(len(test_images)):
        image = test_images[i]
        image = image.reshape((28, 28))
        image = image.astype('uint8')
        image = Image.fromarray(image)
        image.save('../data/test/{}_{}.png'.format(test_labels[i],i))

    print("Dataset is created for %d devices" %(n_device))


def createNonIIDMnistFLData(n_device):

    if not os.path.exists('../data'):
        os.makedirs('../data')
    folderId = 0
    for _ in range(n_device):
        folderId += 1
        if not os.path.exists('../data/node0' + str(folderId) + '/data_mnist'):
            os.makedirs('../data/node0' + str(folderId) + '/data_mnist')       


    #Data preprocessing
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    # Save the train images.
    for i in range(len(train_images)):
    
        image = train_images[i]
        image = image.reshape((28, 28))
        image = image.astype('uint8')
        image = Image.fromarray(image)
        if train_labels[i] == 0 or train_labels[i] == 1:
            folderId = 1
        elif train_labels[i] == 2 or train_labels[i] == 3:
            folderId = 2
        elif train_labels[i] == 4 or train_labels[i] == 5:
            folderId = 3
        elif train_labels[i] == 6 or train_labels[i] == 7:
            folderId = 4
        elif train_labels[i] == 8 or train_labels[i] == 9:
            folderId = 5
        else:
            print(f'Label undetected {train_labels[i]}')
        image.save('../data/node0{}/data_mnist/{}_{}.png'.format(folderId,train_labels[i],i))
    # ensure 100 sample to all 
    # Save the test images.
    if not os.path.exists('../data/test'):
        os.makedirs('../data/test')
    for i in range(len(test_images)):
        image = test_images[i]
        image = image.reshape((28, 28))
        image = image.astype('uint8')
        image = Image.fromarray(image)
        image.save('../data/test/{}_{}.png'.format(test_labels[i],i))

    print("Dataset is created for %d devices" %(n_device))

if __name__ == "__main__":
    createNonIIDMnistFLData(5)