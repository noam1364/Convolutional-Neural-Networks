import tensorflow as tf
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.initializers import TruncatedNormal
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

#Constant Values
i_shape = (32, 32, 3)
kernel_size = (3,3)
pool_size = (2,2)

b_init = keras.initializers.Constant(value=0.025)
k_init = TruncatedNormal(mean=0.0, stddev=0.1)
dense_k_init = TruncatedNormal(mean=0.0, stddev=0.1)
dense_act = 'elu'
conv_act = 'elu'

#CNN Model
model = Sequential()
model.add(Conv2D(32, kernel_size=kernel_size, input_shape=i_shape,activation= conv_act,padding='same',
use_bias=True,kernel_initializer = k_init,bias_initializer = b_init))
model.add(Conv2D(32, kernel_size=kernel_size,activation= conv_act,padding='same',
use_bias=True,kernel_initializer = k_init,bias_initializer = b_init))
model.add(MaxPooling2D(pool_size=pool_size,strides = 2))
model.add(Dropout(0.15))
model.add(Conv2D(64,kernel_size=kernel_size,activation= conv_act,padding='same',
use_bias=True,kernel_initializer = k_init,bias_initializer = b_init))
model.add(Conv2D(64,kernel_size=kernel_size,activation= conv_act,padding='same',
use_bias=True,kernel_initializer = k_init,bias_initializer = b_init))
model.add(MaxPooling2D(pool_size=pool_size,strides = 2))
model.add(Dropout(0.25))
model.add(Flatten()) 
model.add(Dense(128, activation=dense_act,use_bias=True,
kernel_initializer = dense_k_init,bias_initializer = 'zeros'))
model.add(Dropout(0.4))
model.add(Dense(128, activation=dense_act,use_bias=True,
kernel_initializer = dense_k_init,bias_initializer = 'zeros'))
model.add(Dropout(0.5))
model.add(Dense(10,activation=tf.nn.softmax,use_bias = True,
bias_initializer = 'zeros',kernel_initializer = dense_k_init))

i = input('Initiate new model?y/n\n')
if i =='y':
	model.save('Model1.h5')
	input('Model Saved')