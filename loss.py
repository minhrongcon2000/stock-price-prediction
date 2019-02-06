import numpy as np

def crossentropy(pred,label):
	return np.sum(-label*np.log(pred)-(1-label)*np.log(1-pred))

def mse(pred,label):
	return (1/2)*np.sum((pred-label)**2)