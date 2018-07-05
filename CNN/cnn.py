#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 09 13:06:09 2018

@author: keyur-r
"""

# CNN classifier

#import image_preprocessing as imgproc

# Building architecture of our CNN classifier
import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step - 1 Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu' ))

# Step - 2 Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2) ))

# Step - 3 Flattening
classifier.add(Flatten())

# Step - 4 Full connection -> First layer input layer then hidden layer and last softmax layer
classifier.add(Dense(128, activation='relu', kernel_initializer='uniform'))
classifier.add(Dense(10, activation='softmax', kernel_initializer='uniform'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# Image Preprocessing
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator( rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory( 'dataset/training_set', target_size=(64, 64), batch_size=32, class_mode='categorical')

test_set = test_datagen.flow_from_directory( 'dataset/test_set', target_size=(64, 64), batch_size=32, class_mode='categorical')

# Logging the training of models
from keras.callbacks import CSVLogger
csv_logger = CSVLogger('log.csv', append=True, separator=';')

classifier.fit_generator( training_set, steps_per_epoch = 393, epochs = 5, validation_data = test_set, validation_steps = 99, callbacks = [csv_logger])
classifier.save("safedriving_classification.h5")

