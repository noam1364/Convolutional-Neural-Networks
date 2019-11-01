import tensorflow as tf
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

#Constant Values
i_shape = (32, 32, 3)
kernel_size = (3,3)
pool_size = (2,2)

#CNN Model
model = Sequential()
model.add(Conv2D(32, kernel_size=kernel_size, input_shape=i_shape,activation= 'relu',padding='same'))
model.add(Conv2D(32, kernel_size=kernel_size,activation= 'relu',padding='same'))
model.add(MaxPooling2D(pool_size=pool_size,strides = 2))
model.add(Dropout(0.15))
model.add(Conv2D(64,kernel_size=kernel_size,activation= 'relu',padding='same'))
model.add(Conv2D(64,kernel_size=kernel_size,activation= 'relu',padding='same'))
model.add(MaxPooling2D(pool_size=pool_size,strides = 2))
model.add(Dropout(0.25))
model.add(Flatten()) 
model.add(Dense(128, activation='elu',use_bias=True,
kernel_initializer = 'random_uniform',bias_initializer = 'zeros'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='elu',use_bias=True,
kernel_initializer = 'random_uniform',bias_initializer = 'zeros'))
model.add(Dropout(0.5))
model.add(Dense(10,activation=tf.nn.softmax))

i = input('Initiate new model?y/n\n')
if i =='y':
	model.save('Model0.h5')
	input('Model Saved')