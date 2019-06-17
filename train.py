import csv
import argparse
import cv2
import os
from random import shuffle
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

from keras.models import Sequential, load_model
from keras.layers import Flatten,Dense,Lambda,Cropping2D,Conv2D,Dropout,BatchNormalization
from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt

correction = [0,0.25,-0.25]
batch_size=64



def save_hist(lines,file_name):
   steering_data = [float(line[3]) for line in lines[:]]
   fig = plt.hist(steering_data,bins=21)
   plt.xlabel('Steering angle')
   plt.ylabel('Frequency')
   plt.savefig(file_name)



def correct_paths(line,trace):
    for i in range(3):
        filename = line[i].split('\\')[-1]
        current_path = 'data/'+trace+'/IMG/' + filename
        line[i]=current_path


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        images = []
        measurements = []
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            for batch_sample in batch_samples:
                # iterate through center, left and right images
                for i in range(3):
                    image = cv2.imread(batch_sample[i])
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                    images.append(image)
                    # also add flipped images
                    images.append(np.fliplr(image))
                    steering = float(batch_sample[3])+correction[i]
                    measurements.append(steering)
                    measurements.append(-1.0*steering)

            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument(
        'model',
        type=str,
        nargs='?',
        help='Model name',
        default='model.h5'
    )
    parser.add_argument(
        'epochs',
        type=int,
        nargs='?',
        default=5,
        help='Number of epochs'
    )
    args = parser.parse_args()

    lines = []
    for trace in os.listdir('data'):
        with open('data/'+trace+'/driving_log.csv') as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                correct_paths(line,trace)
                if (np.abs(float(line[3]))>0.02):
                     for _ in range(5):
                        lines.append(line)
                elif (np.abs(float(line[3]))>0.2):
                     for _ in range(20):
                        lines.append(line)
                elif (np.abs(float(line[3]))>0.3):
                     for _ in range(40):
                        lines.append(line)
                else:
                    lines.append(line)
    save_hist(lines, 'hist.png')
    train_samples, validation_samples = train_test_split(lines, test_size=0.1)
    
    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)
    
    
    model = Sequential()
    model.add(Cropping2D(cropping=((60, 20), (0, 0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(80,320,3)))
    model.add(Conv2D(24,5,5,subsample=(2, 2),activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(36,5,5,subsample=(2, 2),activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(48,5,5,subsample=(2, 2),activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(64,3,3,activation='relu'))
    model.add(Conv2D(64,3,3,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(100))
    model.add(Dropout(0.2))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.summary()
    model.compile(loss='mse',optimizer='adam')

    if os.path.exists(args.model):
        model = load_model(args.model)
    model_checkpoint = ModelCheckpoint(args.model,verbose=1, save_best_only=True)
    callback_list = [model_checkpoint]

    model.fit_generator(train_generator,steps_per_epoch=np.ceil(len(train_samples)/batch_size),validation_data=validation_generator,validation_steps=np.ceil(len(validation_samples)/batch_size),epochs=args.epochs, verbose=1, callbacks=callback_list)
    
    #model.save(args.model)
    