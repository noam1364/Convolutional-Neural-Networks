import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.models import load_model

#Constant Values
input_shape = (28, 28, 1)
conv_kernel_size = (3,3)
pool_size = (2,2)
relu = tf.nn.relu

#CNN Model
model = Sequential()
#Image Proccecing layers
model.add(Conv2D(32, kernel_size=conv_kernel_size, input_shape=input_shape))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Conv2D(32, kernel_size=conv_kernel_size, input_shape=input_shape))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
#Fully connected layers
model.add(Dense(256, activation=relu))
model.add(Dropout(0.25))
model.add(Dense(256, activation=relu))
model.add(Dropout(0.25))
model.add(Dense(10,activation=tf.nn.softmax))

#Compiling the model and saving it
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',              
metrics=['accuracy'])

i = input('Initiate new model?y/n\n')
if i =='y':
	model.save('Model2.h5')
input('Model Saved')
