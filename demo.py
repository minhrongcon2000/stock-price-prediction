import finance_data
import matplotlib.pyplot as plt 
import numpy as np
from loss import *
from activation import *

np.random.seed(0)

datetime, price = finance_data.get_data('GOOG.csv',True)

#hyperparameters
hidden_unit = 39
output_unit = 1
input_unit = 1
alpha = 0.01
display_step = 1000
train_test_split = 0.7

# split train test data
trainX = price[:int(train_test_split*len(price))]

#parameters
## memory cells
wcx = 2*np.random.random((hidden_unit,input_unit)) - 1
wca = 2*np.random.random((hidden_unit,hidden_unit)) - 1
bc = 2*np.random.random((hidden_unit,1)) - 1

## update gate
wux = 2*np.random.random((hidden_unit,input_unit)) - 1
wua = 2*np.random.random((hidden_unit,hidden_unit)) - 1
bu = 2*np.random.random((hidden_unit,1)) - 1

## forget gate
wfx = 2*np.random.random((hidden_unit,input_unit)) - 1
wfa = 2*np.random.random((hidden_unit,hidden_unit)) - 1
bf = 2*np.random.random((hidden_unit,1)) - 1

## output gate
wox = 2*np.random.random((hidden_unit,input_unit)) - 1
woa = 2*np.random.random((hidden_unit,hidden_unit)) - 1
bo = 2*np.random.random((hidden_unit,1)) - 1

## pred
wya = 2*np.random.random((output_unit,hidden_unit))
by = 2*np.random.random((output_unit,1))

# readability
a = {0: np.zeros((hidden_unit,1))}
pred = {}
c = {0: np.zeros((hidden_unit,1))}
c_tol = {}
da = {}
dc = {}
dc_tol = {}
err = []
j=0
try:
	while True:
		overall = 0.

		dwcx = np.zeros_like(wcx)
		dwca = np.zeros_like(wca)
		dbc = np.zeros_like(bc)
		dwux = np.zeros_like(wux)
		dwua = np.zeros_like(wua)
		dbu = np.zeros_like(bu)
		dwfx = np.zeros_like(wfx)
		dwfa = np.zeros_like(wfa)
		dbf = np.zeros_like(bf)
		dwox = np.zeros_like(wox)
		dwoa = np.zeros_like(woa)
		dbo = np.zeros_like(bo)
		dwya = np.zeros_like(wya)
		dby = np.zeros_like(by)

		# forward pass
		for time in range(1,trainX.shape[0]):
			x = np.expand_dims(np.expand_dims(trainX[time-1],axis=0),axis=0)
			y = np.expand_dims(np.expand_dims(trainX[time],axis=0),axis=0)

			c_tol[time] = tanh(np.dot(wca,a[time-1]) + np.dot(wcx,x) + bc)
			u_gate = sigmoid(np.dot(wua,a[time-1]) + np.dot(wux,x) + bu)
			f_gate = sigmoid(np.dot(wfa,a[time-1]) + np.dot(wfx,x) + bf)
			o_gate = sigmoid(np.dot(woa,a[time-1]) + np.dot(wox,x) + bo)

			c[time] = u_gate*c_tol[time] + f_gate*c[time-1]
			a[time] = o_gate*tanh(c[time])

			pred[time] = np.dot(wya,a[time]) + by

			overall += mse(pred[time],y)

			# Backpropagation
			error = pred[time] - y
			dwya += error.dot(a[time].T)
			dby += np.sum(error,axis=1,keepdims=True)

			da[time] = wya.T.dot(error)
			do_gate = da[time]*tanh(c[time])*sigmoid(np.dot(woa,a[time-1]) + np.dot(wox,x) + bo, deriv=True)
			dwoa += do_gate.dot(a[time-1].T)
			dwox += do_gate.dot(x.T)
			dbo += np.sum(da[time],axis=1,keepdims=True)

			dc[time] = da[time]*o_gate*tanh(c[time],deriv=True)

			du_gate = dc[time]*c_tol[time]*sigmoid(np.dot(wua,a[time-1]) + np.dot(wux,x) + bu, deriv=True)
			dwua += du_gate.dot(a[time-1].T)
			dwux += du_gate.dot(x.T)
			dbu += np.sum(du_gate,axis=1,keepdims=True)

			df_gate = dc[time]*c[time-1]*sigmoid(np.dot(wfa,a[time-1]) + np.dot(wfx,x) + bf, deriv=True)
			dwfa += df_gate.dot(a[time-1].T)
			dwfx += df_gate.dot(x.T)
			dbf += np.sum(df_gate,axis=1,keepdims=True)

			dc_tol[time] = dc[time]*u_gate*tanh(np.dot(wca,a[time-1]) + np.dot(wcx,x) + bc, deriv=True)
			dwca += dc_tol[time].dot(a[time-1].T)
			dwcx += dc_tol[time].dot(x.T)
			dbc += np.sum(dc_tol[time],axis=1,keepdims=True)

		wcx -= alpha*dwcx
		wca -= alpha*dwca
		bc -= alpha*dbc

		wux -= alpha*dwux
		wua -= alpha*dwua
		bu -= alpha*dbu

		wfx -= alpha*dwfx
		wfa -= alpha*dwfa
		bf -= alpha*dbf

		wox -= alpha*dwox
		woa -= alpha*dwoa
		bo -= alpha*dbo

		wya -= alpha*dwya
		by -= alpha*dby
		err.append(overall)
		if j%display_step==0:
			print('Iteration %d, loss %f'%(j,overall))
		j+=1
except KeyboardInterrupt:
	plt.plot(np.arange(1,1+len(err)),err)
	plt.show()
	pred_list = []
	x = np.expand_dims(np.expand_dims(price[0],axis=0),axis=0)
	for time in range(1,datetime.shape[0]):
		c_tol[time] = tanh(np.dot(wca,a[time-1]) + np.dot(wcx,x) + bc)

		u_gate = sigmoid(np.dot(wua,a[time-1]) + np.dot(wux,x) + bu)
		f_gate = sigmoid(np.dot(wfa,a[time-1]) + np.dot(wfx,x) + bf)
		o_gate = sigmoid(np.dot(woa,a[time-1]) + np.dot(wox,x) + bo)

		c[time] = u_gate*c_tol[time] + f_gate*c[time-1]
		a[time] = o_gate*tanh(c[time])

		pred[time] = np.dot(wya,a[time]) + by
		pred_list.append(float(pred[time]))
		x = pred[time]

	plt.figure(figsize=(20,10))
	plt.plot(datetime[1:],finance_data.denomalize(price[1:]),label='actual')
	plt.plot(datetime[1:],finance_data.denomalize(np.array(pred_list)),label='prediction')
	plt.title("Google's Stock Price Prediction")
	plt.legend()
	plt.savefig('pred.png',dpi=300)
	plt.show()














