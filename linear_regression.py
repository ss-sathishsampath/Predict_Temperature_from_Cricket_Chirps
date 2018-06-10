# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 23:57:45 2018
@author: Sathish Sampath(ss.sathishsampath@gmail.com)
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read the input file
df = pd.read_csv("PierceCricketData.csv")



x_data, y_data = (df["Chirps"].values,df["Temp"].values)
X = tf.placeholder(tf.float32, shape=(x_data.size))
Y = tf.placeholder(tf.float32,shape=(y_data.size))
m = tf.Variable(3.0)
c = tf.Variable(2.0)

# Construct a Model
Ypred = tf.add(tf.multiply(X, m), c)


# Create and Run a Session to Visualize the Predicted Line from above Graph
session = tf.Session()
session.run(tf.global_variables_initializer())
pred = session.run(Ypred, feed_dict={X:x_data})

#Initial Prediction vs Datapoints
plt.plot(x_data, pred)
plt.plot(x_data, y_data, 'ro')
plt.xlabel("Number of Chirps every 15 sec")
plt.ylabel("Temperature in Farenhiet")


#Loss Function
nf = 1e-1
loss = tf.reduce_mean(tf.squared_difference(Ypred*nf,Y*nf))


# Defining Gradient Descent Optimizer and implementing it to mimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

# Reinitializing the Global Variables 
session.run(tf.global_variables_initializer())



convergenceTolerance = 0.0001
previous_m = np.inf
previous_c = np.inf

steps = {}
steps['m'] = []
steps['c'] = []

losses=[]

for k in range(100000):
    _, _m , _c,_l = session.run([train, m, c,loss],feed_dict={X:x_data,Y:y_data})
    steps['m'].append(_m)
    steps['c'].append(_c)
    losses.append(_l)
    if (np.abs(previous_m - _m) <= convergenceTolerance) or (np.abs(previous_c - _c) <= convergenceTolerance):
        
        print("Finished by Convergence Criterion")
        print(k)
        print(_l)
        break
    previous_m = _m, 
    previous_c = _c, 
    
session.close() 

# Plotting the Loss against number of steps taken to converge
plt.plot(losses[:])
