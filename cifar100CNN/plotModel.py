import tensorflow as tf
import keras
import graphviz
from keras.utils import plot_model
import matplotlib.pyplot as plt
import globals as g

#Load test data
while True:
	model_path = g.chooseModel()
	history = g.getModelHist(model_path)
	model = keras.models.load_model(model_path)
	plot_model(model,model_path+'.png')
	plt.plot(history['acc'])
	str1 = model_path.split('.')[0]
	str1 = str1.split('\\',1)[1]
	plt.title('Model accuracy:'+str1)
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.show()
