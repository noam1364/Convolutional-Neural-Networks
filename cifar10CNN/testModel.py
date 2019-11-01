import tensorflow as tf
import numpy as np
import keras
import matplotlib.pyplot as plt
import globals as g

#Load model and test data
model_path = g.chooseModel()
model = keras.models.load_model(model_path)
isCustom = input('Use custom test data?y/n\n')
if isCustom == 'y':
	(x_test, y_test) = g.getCustomTestData()
	isCustom = True
else:
	(x_train, y_train), (x_test, y_test) = g.getNormalizedData()
	isCustom = False

#Test model
if not isCustom:
	with tf.device('/gpu:0'):
		loss,acc = model.evaluate(x_test,y_test)
	print('Accuracy:'+str(acc))
	print('Loss:'+str(loss))
cond = True
idx = -1
while cond:
	print(' ')
	if isCustom:
		idx = idx + 1
		if(idx == len(x_test)-1):
			cond = False
	else:
		idx = int(np.random.random()*x_test.shape[0])
		
	image = x_test[idx].reshape(1,32,32,3)
	pred_label = g.cifar10_labels[model.predict(image).argmax()]
	label = g.cifar10_labels[y_test[idx][0]]
	
	print('prediction:'+pred_label+'\n'+'Label:'+label)
	if pred_label == label:
		print('Correct.')
	else:
		print('Wrong.')
	plt.imshow(x_test[idx].reshape(32, 32,3))
	plt.show()
input('\nexit.')