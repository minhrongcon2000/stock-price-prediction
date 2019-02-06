import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

def get_data(filename,normalize=False):
	data = pd.read_csv(filename)

	df = pd.DataFrame(data,columns=['Date','Open','High','Low','Close','Adj Close','Volume'])

	date = ['0']
	prices = []

	for datetime in df['Date']:
		date.append(datetime.split('-')[2])

	for close_price in df['Close']:
		prices.append(float(close_price))

	date = np.array(date)
	prices = np.array(prices)

	if normalize:
		prices = prices/prices[0] - 1

	# date = np.concatenate((['0'],datetime),axis=0)
	prices = np.concatenate(([0.],prices),axis=0)

	return date, prices

def denomalize(data):
	date, prices = get_data('GOOG.csv')
	init_prices = prices[1]

	origin_data = init_prices*(1+data)
	return origin_data