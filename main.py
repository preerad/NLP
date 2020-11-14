# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np
import tensorflow as tf
import pickle
import sklearn
from sklearn.linear_model import LinearRegression

x = np.array([1,2,3,4,5],dtype=float)
x= x.reshape(1,-1)
y = np.array([6,12,18,24,30],dtype=float)
y= y.reshape(1,-1)

#model = tf.keras.Sequential([tf.keras.layers.Dense(1,input_shape=(1,1))])

#model.compile(optimizer='sgd',loss='mean_squared_error',metrics=['accuracy'])

model = LinearRegression()

model.fit(x,y)

pickle.dump(model , open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))

sample = np.array([6]).reshape(1,-1)

