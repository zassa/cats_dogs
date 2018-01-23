#!/usr/bin/env python3
# coding: utf-8

# Note: This code uses keras 2 notation
# see https://faroit.github.io/keras-docs/2.0.2/

print("Importing the Keras libraries and packages...")
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import os

print("Initialising the CNN...")
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu", input_shape=(64, 64, 3)))

# Step 2 - Max Pooling
classifier.add(MaxPooling2D(strides = (2,2)))

# Step 3 - Flatening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units=128, activation="relu"))

classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

# Data augmentation
# random transformations (rotations, flipping, shifting)
train_datagen = ImageDataGenerator(
		# rescale the pixel values from [0, 255] to [0, 1] interval
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

print("importing training data set...")
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                target_size=(64,64),
                                                batch_size=32,
                                                class_mode='binary')
print("importing test data set...")
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')
print("training the network...")
classifier.fit_generator(training_set,
                        steps_per_epoch=8000/32,# number of samples divided by batch size
                        epochs=25,
                        validation_data=test_set,
                        validation_steps=2000/32)



model = '250-samples--25-epochs.h5'
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"models",model)

# Save the model
print('saving the model to\n\t{}'.format(model_path))
classifier.save_weights(model_path)

# Make predictions using a picture of your own cat/dog
print("evaluating the network...")
classifier.evaluate_generator(test_set, 2000/32)
