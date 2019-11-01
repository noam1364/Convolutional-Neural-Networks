import tensorflow as tf
import keras
from keras.models import Sequential
import globals as g

def compileModel0():
	opt = keras.optimizers.rmsprop(lr=0.001,decay=0.001)
	model.compile(optimizer=opt,loss='sparse_categorical_crossentropy',          
	metrics=['accuracy'])

def compileModel1():
	opt = keras.optimizers.Adam(lr=0.00075,beta_1 = 0.9,beta_2 = 0.999,decay = 0.0)
	model.compile(optimizer=opt,loss='sparse_categorical_crossentropy',          
	metrics=['accuracy'])

def compileModel2():
	opt = keras.optimizers.Adam(lr=0.00075,decay=0.0,beta_1 = 0.9,beta_2 = 0.999)
	model.compile(optimizer=opt,loss='sparse_categorical_crossentropy',          
	metrics=['accuracy'])

methods = {'0':compileModel0,'1':compileModel1,'2':compileModel2}
 
model_path = g.chooseModel()
model_num = model_path.split('.')[0][-1]
model = keras.models.load_model(model_path)

methods[model_num]()

i = input('Compile model?y/n\n')
if i =='y':
	model.save(model_path)
	input('Model Compiled')