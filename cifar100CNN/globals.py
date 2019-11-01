import tensorflow as tf
import numpy as np
import keras
from keras.datasets import cifar100

#Constant values
cifar100_labels = ['apple', 'aquarium_fish', 'baby'
,'bear', 'beaver', 'bed', 'bee', 'beetle', 
'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
'worm']
customTestDataPath = 'CustomTestData\\'

#global functions
def getHistPath(model_path):
	return model_path.split('.',1)[0]+'.txt'
	
def getModelHist(model_path):
	h_path = getHistPath(model_path) 
	file = open(h_path,'r')
	data = file.read()
	file.close()
	try:
		data = eval(data)
	except:
		data = {}
	return data
	
def getLogDir(model_path):
	folder = model_path.split('\\',1)[0]
	file = open(folder+r'\logs\logDirs.txt','r')
	logDir = str(file.read())
	file.close()
	file = open(folder+r'\logs\logDirs.txt','w')
	nextLogDir = logDir[0:-1]
	nextLogDir = nextLogDir +str(int(logDir[-1])+1)
	file.write(nextLogDir)
	file.close()
	return logDir

def initLogDir(model_path):
	folder = model_path.split('.',1)[0]
	file = open(folder+r'\logs\logDirs.txt','w')
	newLogDir = logDir[0:-1]
	newLogDir = nextLogDir +str('0')
	file.write(newLogDir)
	file.close()

def getMergedLstItem(a,b,key):	#recives 2 dicts with list values and a key
	#and returns a merged list of the values at the key in each dict
	#only dict 'a' is allowed to be '{}'
	if a!={}:
		list = a[key]
		list2 = b[key]
		for item in list2:
			list.append(item)
	else:
		list = b[key]
	return list
	
def saveHistory(h,model_path):
	file_path = getHistPath(model_path)
	prev_h = getModelHist(model_path)
	curr_h = h.history
	loss = getMergedLstItem(prev_h,curr_h,'loss')
	acc = getMergedLstItem(prev_h,curr_h,'acc')
	hist = {'loss':loss,'acc':acc}
	file = open(file_path,'w')
	file.write(str(hist))
	file.close()
	
def chooseModel():
	str = input('choose Model:')
	return 'Model'+str+'\Model'+str+'.h5'

def getNormalizedData():	#Data pre-proccecing
	(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
	#reshaping training data from 3D to 4D array
	x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
	x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)
	#Normalizing the images
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255
	
	return (x_train,y_train),(x_test,y_test)
	
def getCustomTestData():
	dictFile = open(str(customTestDataPath+'metadata.txt'),'r')
	dict = eval(dictFile.read())
	dictFile.close()
	x_test = []
	y_test = []
	for elem in dict:
		im = io.imread(str(customTestDataPath+elem))
		x_test.append(np.array(im))
		y_test.append([getNumLabel(dict[elem])])
	#Normalize data
	x_test = np.array(x_test)
	x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)
	#Normalizing the images
	x_test = x_test.astype('float32')
	x_test /= 255
	return (x_test,y_test)
	
def getNumLabel(str):	#for cifar10 database
	for i in range(len(cifar10_labels)):
		if cifar10_labels[i] == str:
			return i
	return -1