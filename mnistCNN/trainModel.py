import tensorflow as tf
import keras
import globals as g

#Configuring a training session
config = tf.ConfigProto() 
config.gpu_options.per_process_gpu_memory_fraction = 1
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

#Load model and test data
model_path = g.chooseModel()
try:
	model = keras.models.load_model(model_path)
except:
	print('Model file not found')
(x_train, y_train), (x_test, y_test) = g.getNormalizedData()

#Train model
i='y'
with sess:
	while i =='y':
		epoc = int(input('enter number of epochs:'))
		batch_num = int(input('enter batch size:'))
		with tf.device('/gpu:0'):
			history = model.fit(x=x_train,y=y_train, epochs=epoc,batch_size=batch_num)
		model.save(model_path)
		g.saveHistory(history,model_path)
		i = input('Another training session?y/n\n')
sess.close()