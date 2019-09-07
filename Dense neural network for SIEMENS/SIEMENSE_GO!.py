# -*- coding: utf-8 -*-
"""
SIEMENS, Go!
训练和生成：B->C
"""

# -*- coding: utf-8 -*-
"""
SIEMENS, Yes!!!
"""
import os
import csv
import numpy as np
import pandas as pd
from pandas import read_csv
import math
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Reshape, MaxPooling1D, Conv1D, GlobalAveragePooling1D, Dropout, Dense, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD

'''
train_data:total:6000-48=5952
64 * 80 = 5120 for train
6000-5120 = 832 for validation
'''

num_train = 5120
batch_size = 64
node_output = 4
TRAIN = False
data_path = 'siemens.csv'
test_path = 'first_round_testing_data.csv'

def convert2oneHot(index,Lens):
    hot = np.zeros((Lens,))
    hot[int(index)] = 1
    return(hot)

def train_and_val_datagen(data_path, batch_size, whether_train):
	data = read_csv(data_path,
				delimiter=',',
				header=0)
	data = np.array(data)[: , :]
	print("shape of the data feeded:",data.shape)
	if whether_train:
		data = np.array(data)[:num_train]
		steps = math.ceil(len(data) / batch_size)
	else:
		data = np.array(data)[num_train:]
		steps = math.ceil(len(data) / batch_size)
	while True:
		for i in range(steps):
			batch_list = data[i*batch_size : i*batch_size + batch_size]
			np.random.shuffle(batch_list)
			batch_x = np.array([file for file in batch_list[:,13:20]])
			batch_y = np.array([convert2oneHot(label,4) for label in batch_list[:,-1]])
			yield batch_x, batch_y

def test_datagen(path, batch_size):
	data = read_csv(path,
				delimiter=',',
				header=0)
	test = np.array(data)[:,11:18]
	steps = math.ceil(len(test)/batch_size)
	while True:
		for i in range(steps):
			batch_list = test[i*batch_size : i*batch_size+batch_size]
			batch_xs = np.array([file for file in batch_list])
			yield batch_xs

	

num_param = 7

#model needs modify
def Model(num_classes):
    model = Sequential()
    model.add(Reshape((num_param, ), input_shape=(num_param, )))
    model.add(Dense(256, activation='relu'))#input_shape=(num_1st_layer_node, )
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    return model
'''
A回归B时，只需Dense512，B分类C时，需要Dense512+Dense256，并且注意修改loss
'''


if __name__ == "__main__":
	if TRAIN == True:
		
		
		trainflow_generator = train_and_val_datagen(data_path,batch_size,True)
		valflow_generator =  train_and_val_datagen(data_path,batch_size,False)
		
		model = Model(num_classes = node_output)
		
		#lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch/20))
		#optimizer = tf.keras.optimizers.SGD(lr=8e-3, momentum=0.9)
		optimizer = Adam(5e-4)
		
		model.compile(loss = "categorical_crossentropy",
				optimizer = optimizer, metrics = ['accuracy'])
		model.summary()
		
		#callbacks = myCallback()
		history = model.fit_generator(
				trainflow_generator,
				validation_data = valflow_generator,
				steps_per_epoch = num_train//batch_size, # 10=640/batch_size
				epochs = 100,
				validation_steps = (6000 - num_train)//batch_size,
				verbose = 2,
				#callbacks=[lr_schedule]
				)
		#plt.semilogx(history.history["lr"], history.history["loss"])
		#plt.axis([1e-8, 1e-3, 0, 60])
		
		acc = history.history['accuracy']
		val_acc = history.history['val_accuracy']
		loss = history.history['loss']
		val_loss = history.history['val_loss']
		
		epochs = range(len(acc))

		plt.plot(epochs, acc, 'r', label='training accuracy')
		plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
		plt.title('Training and validation accuracy')
		plt.legend(loc=0)
		plt.figure()
		
		plt.show()
	
	else:
		testflow_generator = test_datagen(test_path,batch_size)
		
		
		prediction = model.predict_generator(generator=testflow_generator,steps=math.ceil(6000/batch_size),verbose=1)
		print(prediction.shape)
		np.savetxt('C_for_test.csv', prediction, delimiter = ',')  
		
		
		
		
		
		















