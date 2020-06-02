#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ArthurYang
"""
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import datasets,models,layers
import os
import time


data = tf.keras.datasets.cifar10

class_names = ['airplane','automobile','bird','cat','deer','dog',
               'frog','horse','ship','truck']

(x_train_1, y_train_1), (x_test_1, y_test_1) = data.load_data()
x_train_1 = x_train_1 / 255.0
y_train_1 = y_train_1 / 255.0

train_image = os.listdir("./train/")
test_image = os.listdir("./test/")

train_num = len(train_image)
test_num = len(test_image)



trainlabel = pd.read_csv("trainLabels.csv")

y_train = np.array([[class_names.index(trainlabel.label[i])] for i in range(train_num)])

def load_image(image):
    size = (32, 32)
    img = tf.io.read_file(image)
    img = tf.image.decode_png(img)
    img = tf.image.resize(img, size) / 255.0
    return(img)

         

print("Decoding image...")

t0 = time.process_time()
train_raw = tf.Variable([load_image("./train/"+ str(i + 1) + ".png") for i in range(train_num)])
x_train = tf.convert_to_tensor(train_raw)


t1 = time.process_time()
print("Done with train set, spend %f s" %(t1 - t0))
print("Decoding test set")


test_raw = tf.Variable([load_image("./test/"+ str(i + 1) + ".png") for i in range(test_num)])
x_test = tf.convert_to_tensor(test_raw)

t2 = time.process_time()
print("Done with test set, spend %f s" %(t2 - t1))


model = tf.keras.models.Sequential([
                        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
                        tf.keras.layers.MaxPooling2D(2, 2),
                        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                        tf.keras.layers.MaxPooling2D(2, 2),
                        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                        tf.keras.layers.Flatten(),
                        tf.keras.layers.Dense(128, activation='relu'),
                        tf.keras.layers.Dense(10, activation='softmax')])



model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=20, validation_data=(x_test_1, y_test_1))

prediction = model.predict(x_test)

name = [class_names[np.argmax(prediction[i])] for i in range(len(prediction))]

id1 = []

for i in range(len(prediction)):
    id1.append(i + 1)
    
result = pd.DataFrame({'id':id1, 'label':name})

result.to_csv('submission.csv',index=False)


