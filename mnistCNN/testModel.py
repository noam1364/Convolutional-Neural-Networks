import tensorflow as tf
import numpy as np
import keras
from keras.utils import plot_model
import matplotlib.pyplot as plt
import globals as g

#Load model and test data
model_path = g.chooseModel()
model = keras.models.load_model(model_path)
(x_train, y_train), (x_test, y_test) = g.getNormalizedData()

#Test model
loss,acc = model.evaluate(x_test,y_test)
print('Accuracy:'+str(acc))
print('Loss:'+str(loss))

while True:
	next = input(' ')
	idx = int(np.random.random()*10000)
	image = x_test[idx].reshape(1,28,28,1)
	pred = model.predict(image)
	print('prediction:'+str(pred.argmax())+'\n'+'test image #'+str(idx))
	plt.imshow(x_test[idx].reshape(28, 28),cmap='Greys')
	plt.show()