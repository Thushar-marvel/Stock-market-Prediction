# pip install streamlit fbprophet yfinance plotly
import streamlit as st
# import sklearn
from datetime import date

import yfinance as yf
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout

import datetime

from statsmodels.tsa.holtwinters import ExponentialSmoothing

from plotly import graph_objs as go


START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")


st.title('Stock Prediction Web App')

# stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
stocks = ('AAPL' ,"GOOG" ,"TSLA" ,"AMZN")
selected_stock = st.selectbox('Select dataset for prediction', stocks)




@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')


# Predict forecast with Prophet.
df_train = data[['Date','Close']]
# df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
lstmModel = tf.keras.models.load_model("LSTM_StockPred.h5")
if stocks == "AAPL":
	st.subheader('APPLE Stock')
if stocks == 'AMZN':
	st.subheader('AMAZON Stock')
	lstmModel = tf.keras.models.load_model("modelLSTMamz.h5")
if stocks == "GOOG":
	st.subheader('GOOGLE Stock')
	lstmModel = tf.keras.models.load_model("modelLSTMgog.h5")

if stocks == "TSLA":
	st.subheader('TESLA Stock')
	lstmModel = tf.keras.models.load_model("modelLSTMtsl.h5")

latestData = pd.DataFrame(df_train)
latestData = latestData.tail(250)

st.subheader('Raw data')
st.write(data.tail())

def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	# fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)


plot_raw_data()

def preprocessData(aplData):

	def ewm(data, column, window, span):
		result = [0]
		for i in pd.DataFrame.rolling(data[column], window):
			result.append(np.mean([k for k in i.ewm(span=span, adjust=False).mean()]))
		return result[:-1]

	def autocorr(array, column, window, lag=2):
		w = window + lag
		result = [0] * (w)
		print(array.shape[0])
		for i in range(w, array.shape[0]):
			data = array[column][i - w:i]
			d = data
			y_bar = np.mean(data)
			denominator = sum([(i - y_bar) ** 2 for i in data])
			lagData = [i - y_bar for i in d][lag:]
			actualData = [i - y_bar for i in d][:-lag]
			numerator = sum(np.array(lagData) * np.array(actualData))
			result.append((numerator / denominator))

		return result

	def doubleExSmoothing(array, column, window, trend):
		result = [0] * (window)
		for i in range(window, array.shape[0]):
			data = array[column][i - window:i]
			values = ExponentialSmoothing(data, trend=trend).fit().fittedvalues
			d = [i for i in values.tail(1)]
			result.append(d[0])

		return result


	aplData['Close'] = aplData['Close'].shift(-1)
	moving_AverageValues = [10, 20, 50]
	for i in moving_AverageValues:
		column_name = "MA_%s" % (str(i))
		aplData[column_name] = pd.DataFrame.rolling(aplData['Close'], i).mean().shift(1)
	aplData['5_day_std'] = aplData['Close'].rolling(window=5).std().shift(1)
	aplData['Daily Return'] = aplData['Close'].pct_change().shift(1)
	aplData['SD20'] = aplData.Close.rolling(window=20).std().shift(1)
	aplData['Upper_Band'] = aplData.Close.rolling(window=20).mean().shift(1) + (aplData['SD20'] * 2)
	aplData['Lower_Band'] = aplData.Close.rolling(window=20).mean().shift(1) - (aplData['SD20'] * 2)
	aplData['Close(t-1)'] = aplData.Close.shift(periods=1)
	aplData['Close(t-2)'] = aplData.Close.shift(periods=2)
	aplData['Close(t-5)'] = aplData.Close.shift(periods=5)
	aplData['Close(t-10)'] = aplData.Close.shift(periods=10)
	aplData['EMA_10'] = ewm(aplData, "Close", 50, 10)
	aplData['EMA_20'] = ewm(aplData, "Close", 50, 20)
	aplData['EMA_50'] = ewm(aplData, "Close", 50, 50)
	aplData['MACD'] = aplData['EMA_10'] - aplData['EMA_20']
	aplData['MACD_EMA'] = ewm(aplData, "MACD", 50, 9)
	aplData['ROC'] = ((aplData['Close'].shift(1) - aplData['Close'].shift(10)) / (aplData['Close'].shift(10))) * 100
	funct = lambda x: pd.Series(extract_date_features(x))
	aplData[['Day', 'DayofWeek', 'DayofYear', 'Week', 'Is_month_end', 'Is_month_start', 'Is_quarter_end',
			 'Is_quarter_start', 'Is_year_end', 'Is_year_start', 'Is_leap_year', 'Year', 'Month', "Is_Monday",
			 "Is_Tuesday", "Is_Wednesday", "Is_Thursday", "Is_Friday"]] = aplData["Date"].apply(funct)
	aplData['AutoCorr_1'] = autocorr(aplData, 'Close', 10, 1)
	aplData['AutoCorr_2'] = autocorr(aplData, 'Close', 10, 2)
	aplData['HWES2_ADD'] = doubleExSmoothing(aplData, 'Close', 50, 'additive')
	aplData['HWES2_MUL'] = doubleExSmoothing(aplData, 'Close', 50, 'multiplicative')

	aplData = aplData.iloc[:-1]
	aplData = aplData.tail(50)
	aplData.reset_index(inplace=True)
	aplData = aplData.drop(['index'], axis=1)
	return aplData



def extract_date_features(date_val):
	Day = date_val.day
	DayofWeek = date_val.dayofweek
	Dayofyear = date_val.dayofyear
	Week = date_val.week
	Is_month_end = date_val.is_month_end.real
	Is_month_start = date_val.is_month_start.real
	Is_quarter_end = date_val.is_quarter_end.real
	Is_quarter_start = date_val.is_quarter_start.real
	Is_year_end = date_val.is_year_end.real
	Is_year_start = date_val.is_year_start.real
	Is_leap_year = date_val.is_leap_year.real
	day = date_val.weekday()
	Is_Monday = 1 if day == 0 else 0
	Is_Tuesday = 1 if day == 1 else 0
	Is_Wednesday = 1 if day == 2 else 0
	Is_Thursday = 1 if day == 3 else 0
	Is_Friday = 1 if day == 4 else 0
	Year = date_val.year
	Month = date_val.month

	return Day, DayofWeek, Dayofyear, Week, Is_month_end, Is_month_start, Is_quarter_end, Is_quarter_start, Is_year_end, Is_year_start, Is_leap_year, Year, Month, Is_Monday, Is_Tuesday, Is_Wednesday, Is_Thursday, Is_Friday

def extractFeatures(lastFeatures, y_train, date):
	'''Function used to extract input features based on previous close price and date value
     lastFeatures - last few values (features)
     y_train - close price values
     date - current date
     return - all input features '''



	def autocorr(array, window, lag=2):
		w = window + lag

		data = array
		d = data
		y_bar = np.mean(data)
		denominator = sum([(i - y_bar) ** 2 for i in data])
		lagData = [i - y_bar for i in d][lag:]
		actualData = [i - y_bar for i in d][:-lag]
		numerator = sum(np.array(lagData) * np.array(actualData))

		return numerator / denominator

	def doubleExSmoothing(array, trend):
		data = array
		values = ExponentialSmoothing(data, trend=trend).fit().fittedvalues
		d = [i for i in values.tail(1)]
		#     result.append( d[0])

		return d[0]

	currentData = {i: {None} for i in lastFeatures.columns}
	if lastFeatures.shape[0] > 50 or True:
		lastFeatures = lastFeatures.tail(50)
		lastValue = lastFeatures.iloc[-1]

		end_date = date

		day = pd.to_datetime(end_date).weekday()
		if day <= 4:

			currentData['MA_10'] = y_train.tail(10).mean()
			currentData['MA_20'] = y_train.tail(20).mean()
			currentData['MA_50'] = y_train.tail(50).mean()
			currentData['5_day_std'] = y_train.tail(5).std()
			#             currentData['SD20'] = y_train.tail(20).rolling(window=20).std().iloc[-1]
			currentData['SD20'] = y_train.tail(20).std()
			currentData['Daily Return'] = y_train.tail(2).pct_change().iloc[-1]
			currentData['Upper_Band'] = (y_train.tail(20).mean() + (currentData['SD20'] * 2))
			currentData['Lower_Band'] = (y_train.tail(20).mean() - (currentData['SD20'] * 2))
			#             print(currentData['Upper_Band'])
			currentData['Close(t-1)'] = y_train.iloc[-1]
			currentData['Close(t-2)'] = y_train.iloc[-2]
			currentData['Close(t-5)'] = y_train.iloc[-5]
			currentData['Close(t-10)'] = y_train.iloc[-10]

			#             aplData['EMA_10'] = ewm(aplData, "Close",50,10)
			currentData['EMA_10'] = np.mean(y_train.tail(50).ewm(span=10, adjust=False).mean())
			currentData['EMA_20'] = np.mean(y_train.tail(50).ewm(span=20, adjust=False).mean())


			currentData['EMA_50'] = np.mean(y_train.tail(50).ewm(span=50, adjust=False).mean())
			currentData['MACD'] = currentData['EMA_10'] - currentData['EMA_20']
			currentData['MACD_EMA'] = np.mean(lastFeatures['MACD'].tail(50).ewm(span=9, adjust=False).mean())
			#             currentData['MACD_EMA'] = lastFeatures['MACD'].tail(50).ewm(span=9, adjust=False).mean().iloc[-1]
			currentData['ROC'] = ((y_train.iloc[-1] - y_train.iloc[-10]) / (y_train.iloc[-10])) * 100
			result = list(extract_date_features(end_date))

			for i, v in enumerate(
					['Day', 'DayofWeek', 'DayofYear', 'Week', 'Is_month_end', 'Is_month_start', 'Is_quarter_end',
					 'Is_quarter_start', 'Is_year_end', 'Is_year_start', 'Is_leap_year', "Year", 'Month', "Is_Monday",
					 "Is_Tuesday", "Is_Wednesday", "Is_Thursday", "Is_Friday"]):
				currentData[v] = result[i]
			currentData["AutoCorr_1"] = autocorr(y_train.tail(11), 10, lag=1)
			currentData["AutoCorr_2"] = autocorr(y_train.tail(12), 10, lag=2)
			currentData["HWES2_MUL"] = doubleExSmoothing(y_train.tail(50), 'multiplicative')
			currentData['HWES2_ADD'] = doubleExSmoothing(y_train.tail(50), 'additive')
		else:
			pass


	return currentData

def predictFuture(Nday,X_train,y_train):
	lastFeatures, lastPriceValues = pd.DataFrame(X_train).tail(50), y_train.tail(50)
	date = "{}/{}/{}".format(int(lastFeatures.iloc[-1]['Month']), int(lastFeatures.iloc[-1]['Day']),
							 int(lastFeatures.iloc[-1]['Year']))
	startDate = pd.to_datetime(date) + datetime.timedelta(days=1)
	lastDate = date
	predValues = []
	totalDates = []
	i = 1
	fi = []
	while i <= Nday:
		end_date = pd.to_datetime(date) + datetime.timedelta(days=1)
		date = end_date
		day = pd.to_datetime(end_date).weekday()
		if day <= 4:
			i += 1
			lastDate = end_date
			if i == 2:
				startDate = end_date
			currentData = extractFeatures(lastFeatures, lastPriceValues, end_date)

			df = {k: [v] for k, v in currentData.items()}
			df["Date"] = end_date
			totalDates.append(end_date)

			df = pd.DataFrame(df)


			df3 = pd.concat([lastFeatures, df], ignore_index=True)

			inputFeature = df3.drop(["Date"],axis = 1).iloc[-1]

			inpu = np.array([i for i in inputFeature])

			inpu = sX_val.transform(inpu.reshape(1, -1))


			inpu = inpu.reshape(1, X_train.shape[1]-1, 1)


			pred = lstmModel.predict(inpu, verbose=0)


			pred = sY_val.inverse_transform(pred.reshape(-1, 1))

			predValues.append(pred[0][0])
			lastFeatures = df3.tail(50)
			lastPriceValues = list(lastPriceValues)
			lastPriceValues.append(pred[0][0])
			lastPriceValues = pd.Series(lastPriceValues)
			lastPriceValues = lastPriceValues.tail(50)


	df3["pred"] = lastPriceValues
	predictions = pd.DataFrame(data = [np.array(totalDates),np.array(predValues)]).T
	predictions.columns = ['Date', 'pred']

	return predictions

def modelTrain(xtrain,ytrain,epochs = 50):
	modelLSTM = Sequential()
	modelLSTM.add(LSTM(units=100,return_sequences=True, input_shape=(xtrain.shape[1],1)))
	modelLSTM.add(Dropout(0.2))
	modelLSTM.add(LSTM(units=100, return_sequences=True))
	modelLSTM.add(Dropout(0.2))
	modelLSTM.add(LSTM(units=50))
	modelLSTM.add(Dropout(0.2))
	modelLSTM.add(Dense(units=1))	
	modelLSTM.compile(loss = 'mae', optimizer='adam')
	modelLSTM.fit(xtrain, ytrain, epochs=epochs, batch_size=16)
	return modelLSTM
    

n_days = st.slider('Please enter number of days for prediction:', 0, 365)
period = n_days
Ndays = n_days
if Ndays>0:
	predict_state = st.text('Processing Data, Please wait...')
	inputData = preprocessData(latestData)

	y_test = inputData["Close"].tail(100)
	x_test = inputData.drop(["Close"], axis=1).tail(100)
	xtrain = inputData.drop(["Close"], axis=1)[0:-100,:]
	ytrain = inputData["Close"][0:-100,:]
	sX_train = MinMaxScaler(feature_range=(0, 1))
	sY_train = MinMaxScaler(feature_range=(0, 1))

	X_train = sX_train.fit_transform(np.array(xtrain.drop(["Date"], axis=1))).reshape(xtrain.shape[0],
																					   xtrain.shape[1] - 1, 1)
	Y_train = sY_train.fit_transform(np.array(ytrain).reshape(-1, 1)).reshape(ytrain.shape[0], )


	sX_val = MinMaxScaler(feature_range=(0, 1))
	sY_val = MinMaxScaler(feature_range=(0, 1))
	X_valLSTM = sX_val.fit_transform(np.array(x_test.drop(["Date"], axis=1))).reshape(x_test.shape[0],
																					   x_test.shape[1] - 1, 1)
	y_valLSTM = sY_val.fit_transform(np.array(y_test).reshape(-1, 1)).reshape(y_test.shape[0], )
	predict_state.text(' Done!,Click on predict')
	predict_state.text("Please click the below button, if wanted to update the model by re training in with latest data")
	if st.button("Re-Train Model"):
		predict_state.text(' Training')
		lstmModel = modelTrain(X_train,Y_train)
		predict_state.text(' Trained')


	if st.button("Predict"):
		predict_state.text("Please wait............")

		predictions = predictFuture(Ndays,x_test,y_test)
# 		predictions.iloc[0] =  df_train.iloc[-1]
#                 predictions["Date"].iloc[0] = data["Date"].iloc[-1]
# 		predictions["pred"].iloc[0] = data["Close"].iloc[-1]
		predict_state.text("Done")
		# Show and plot forecast
		st.subheader('Forecasted data')
		st.write(predictions)
# 		st.write(predictions.iloc[0])

		st.write(f'Forecast plot for {Ndays} days')

		fig = go.Figure()
		fig.add_trace(go.Scatter(x=predictions['Date'], y=predictions['pred'],name="Prediction"))
		fig.add_trace(go.Scatter(x=data['Date'].tail(1000), y=data['Close'].tail(1000), name="stock_close"))
		st.plotly_chart(fig)

		st.write("Percentage gain or loss for next {} days prediction ".format(Ndays))
		st.write("Predict price is {} ".format(predictions["pred"].iloc[-1]))
		profit = ((predictions["pred"].iloc[-1] - data['Close'].iloc[-1]  ) / data['Close'].iloc[-1])* 100
		st.write("{} percentage".format(np.round(profit,3)))
		


