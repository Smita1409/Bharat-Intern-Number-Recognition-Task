from os import access
import pandas as  pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from tensorflow.python.keras import activations
from tensorflow.python.keras.activations import softmax
data=tf.keras.datasets.mnist
from tensorflow.python.keras.metrics import accuracy

(x_train,y_train),(x_test,y_test)=data.load_data()

x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)
model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(units=128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10,activation=tf.nn.softmax))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train , epochs=3)

loss,accuracy=model.evaluate(x_test,y_test)
print(accuracy)
print(loss)

for x in range(1,5):
    img=cv.imread(f'{x}.png')[:,:,0]
    img=np.invert(np.array([img]))
    prediction=model.predict(img)
    print("--------------------")
    print("The perdicted output is : ",np.argmax(prediction))
    print("--------------------")
    plt.imshow(img[0],cmap=plt.cm.binary)
    plt.show()