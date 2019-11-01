import tensorflow as tf
import keras
import globals as g
from keras.backend.tensorflow_backend import set_session

#Load model and test data
model_path = g.chooseModel()
try:
	model = keras.models.load_model(model_path)
except:
	print('Error Loading Model')
(x_train, y_train), (x_test, y_test) = g.getNormalizedData()

#Callbacks
tb = keras.callbacks.TensorBoard(log_dir=g.getLogDir(model_path), histogram_freq=1, batch_size=0,
 write_graph=True, write_grads=False, write_images=True, embeddings_freq=0,
 embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='batch')

#Train model
i='y'
while i =='y':
	epoc = int(input('enter number of epochs:'))
	batch_num = int(input('enter batch size:'))
	tb.batch_size = batch_num
	with tf.device('/gpu:0'):
		history = model.fit(x=x_train,y=y_train, epochs=epoc,batch_size=batch_num,
		validation_split = 0.1,callbacks = [tb])
	model.save(model_path)
	g.saveHistory(history,model_path)
	i = input('Another training session?y/n\n')