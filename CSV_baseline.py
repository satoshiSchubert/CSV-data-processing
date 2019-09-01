'''
crisprhhx
'''
import os
import csv
import numpy as np
import pandas
from pandas import read_csv
import math
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Reshape, MaxPooling1D, Conv1D, GlobalAveragePooling1D, Dropout, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
Long = 792
Lens = 640
path = 'zhouchen/train.csv'
test_path = 'zhouchen/test.csv'
batch_size = 64
num_classes = 10
Train = 0

'''
关于yield函数的用法https://blog.csdn.net/mieleizhi0522/article/details/82142856

convert2onehot:[1,  ->   [[0,1,0,0,...],
				2,        [0,0,1,0,...],
				3]        [0,0,0,1,...]]

tensorflow的fit_generator中的generator参数应该得是一个迭代器，
而不是一个以return结尾的普通函数。
因为data是一种“流”，得按顺序不断产生。
'''

def convert2oneHot(index,Lens):
    hot = np.zeros((Lens,))
    hot[int(index)] = 1
    return(hot)
	
def train_and_val_datagen(path,batch_size,whether_train):
	data = read_csv(path,
				delimiter=',',
				header=0)
	if whether_train:
		data = np.array(data)[:Lens]
		steps = math.ceil(len(data) / batch_size)
	else:
		data = np.array(data)[Lens:]
		steps = math.ceil(len(data) / batch_size)
	while True:
		for i in range(steps):
			batch_list = data[i*batch_size : i*batch_size + batch_size]
			np.random.shuffle(batch_list)
			batch_x = np.array([file for file in batch_list[:,1:-1]])#不是很清楚为什么这里要file for file训练才不会出错；初步猜测是因为batch_y也有这一过程，可能要两两匹配？
			#batch_x = batch_list[:, 1:-1]
			batch_y = np.array([convert2oneHot(label,10) for label in batch_list[:,-1]])
			yield batch_x, batch_y
		
		
def test_datagen(path, batch_size):
	data = read_csv(path,
				delimiter=',',
				header=0)
	test = np.array(data)[:,1:]
	steps = math.ceil(len(test)/batch_size)
	while True:
		for i in range(steps):
			batch_list = test[i*batch_size : i*batch_size+batch_size]
			batch_xs = batch_list
			yield batch_xs


TIME_PERIODS = 6000

#model needs modify
def Model(num_classes, input_shape=(TIME_PERIODS,)):
    model = Sequential()
    model.add(Reshape((TIME_PERIODS, 1), input_shape=input_shape))
    model.add(Conv1D(16, 8,strides=2, activation='relu',input_shape=(TIME_PERIODS,1)))
    model.add(Conv1D(16, 8,strides=2, activation='relu',padding="same"))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(64, 4,strides=2, activation='relu',padding="same"))
    model.add(Conv1D(64, 4,strides=2, activation='relu',padding="same"))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(256, 4,strides=2, activation='relu',padding="same"))
    model.add(Conv1D(256, 4,strides=2, activation='relu',padding="same"))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(512, 2,strides=1, activation='relu',padding="same"))
    model.add(Conv1D(512, 2,strides=1, activation='relu',padding="same"))
    model.add(MaxPooling1D(2))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    return model
'''
    class myCallback(tf.keras.callbacks.Callback):
          def on_epoch_end(self, epoch, logs={}):
            if (logs.get('acc')>0.999) :
                print("\nReached 99.9% accuracy so cancelling training!")
                self.model.stop_training = True
'''

if __name__ == "__main__":
	if Train == True:
		
		trainflow_generator = train_and_val_datagen(path,batch_size,True)
		valflow_generator =  train_and_val_datagen(path,batch_size,False)
		
		model = Model(num_classes = num_classes)
		
		opt = Adam(2e-4)
		model.compile(loss = "categorical_crossentropy",
				optimizer = opt, metrics = ['accuracy'])
		model.summary()
		
		#callbacks = myCallback()
		history = model.fit_generator(
				trainflow_generator,
				validation_data = valflow_generator,
				steps_per_epoch = Lens//batch_size, # 10=640/batch_size
				epochs = 50,
				validation_steps = (Long - Lens)//batch_size,
				verbose = 1,
				#callbacks = [callbacks]
				)

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
		
		pres = model.predict_generator(generator=testflow_generator,steps=math.ceil(528/batch_size),verbose=1)
		ohpres = np.argmax(pres,axis=1)#取出最大值所对应的索引
		
		answer = open('answer.csv', 'w')
		writer = csv.writer(answer)
		writer.writerow(['id', 'label'])
		count = 1
		
		for i in ohpres:
			writer.writerow([count, i])
			count+=1











