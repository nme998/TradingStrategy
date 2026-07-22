import math
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
import datetime
import joblib
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests

def predict_stock(Stock):
  search_stock = Stock
  end_date = datetime.datetime.now() 
  start_date = end_date - datetime.timedelta(days = 365 * 5)
  stock_data = yf.download(search_stock, start=start_date.date(), end=end_date.date())
  stock_data.head()

  sentiment_values = get_combined_news("apple")

  close_prices = stock_data['Close']
  values = close_prices.values
  training_data_len = math.ceil(len(values)* 0.8)

  scaler = MinMaxScaler(feature_range=(0,1))
  scaled_data = scaler.fit_transform(values.reshape(-1,1))

  train_data = scaled_data[0: training_data_len, :]
  sentiment_train = sentiment_values[0: training_data_len, :]

  x_train = []
  y_train = []
  z_train = []
  x_sentiment = []

  n_lookback = 60
  n_forecast = 30

  for i in range(60, len(train_data)):
      x_train.append(train_data[i-60:i, 0])
      y_train.append(train_data[i, 0])
      z_train.append(scaled_data[i:i+n_forecast, 0])#training with the 30 day future forecast
      x_sentiment.append(sentiment_train[i-60:i, 0])
      
  x_train, y_train, z_train, x_sentiment = np.array(x_train), np.array(y_train), np.array(z_train), np.array(x_sentiment)

  x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
  x_sentiment = np.reshape(x_sentiment, (x_sentiment.shape[0], x_sentiment.shape[1], 1))
  print("XTRAIN", x_train.shape)
  print("XSENTIMENT", x_sentiment.shape)

  x_train = np.concatenate([x_train, x_sentiment], axis=-1)#combining closing price and sentiment features 

  test_data = scaled_data[training_data_len-60: , : ]
  sentiment_test = sentiment_values[training_data_len-60: , : ]
  x_test = []
  x_sent_test = []
  y_test = values[training_data_len:]


  for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    x_sent_test.append(sentiment_test[i-60:i, 0])

  x_test, x_sent_test = np.array(x_test), np.array(x_sent_test)
  x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
  x_sent_test = np.reshape(x_sent_test, (x_sent_test.shape[0], x_sent_test.shape[1], 1))

  x_test = np.concatenate([x_test, x_sent_test], axis=-1)#combining closing price and sentiment features 

  filename = 'stock_prediction_model.sav'
  #'''
  model = keras.Sequential()
  model.add(layers.LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
  model.add(layers.LSTM(100, return_sequences=False))
  model.add(layers.Dense(25))
  model.add(layers.Dense(1))
  model.summary()

  model.compile(optimizer='adam', loss='mean_squared_error', metrics = ['accuracy'])
  model.fit(x_train, y_train, batch_size= 1, epochs=3)

  joblib.dump(model, filename)
  #'''
  loaded_model = joblib.load(filename)

  predictions = loaded_model.predict(x_test)
  predictions = scaler.inverse_transform(predictions)
  forecastPlot = predictions[:,-1:]
  np.transpose(forecastPlot)
  rmse = np.sqrt(np.mean(forecastPlot - y_test)**2)
  rmse

  results = loaded_model.evaluate(x_test, y_test)
  print(loaded_model.metrics_names)
  print(results)


  
  data = stock_data.filter(['Close'])
  train = data[:training_data_len]
  validation = data[training_data_len:]
  validation['Predictions'] = predictions
  recommendation = model_recommendation(predictions[-30,0], predictions[-1,0])
  stock_plot = plt.figure(figsize=(4,2))
  plt.title('Model')
  plt.xlabel('Date')
  plt.ylabel('Close Price USD ($)')
  plt.plot(train)
  plt.plot(validation[['Close', 'Predictions']])
  plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
  plt.show()


  return stock_plot,recommendation

def model_recommendation(start_value, end_value):
  change_percentage = ((start_value - end_value)/start_value)*100
  if change_percentage >= 1:
    recommendation = 'BUY'
  elif change_percentage <= -1:
    recommendation = 'SELL'
  else:
    recommendation = 'HOLD'

  return recommendation  

def get_combined_news(stock):
  company = "apple"  # replace with your desired company name
  api_key = "2191f98546884e75b8ba4b360a9d6b39"

  # Get headlines for the last 5 days
  timestamp = pd.Timestamp(datetime.datetime(2020,10,10))
  end_date = timestamp.today()
  end_date = end_date - datetime.timedelta(days = 1)
  dates = [end_date - datetime.timedelta(days=i) for i in range(5)]

  # Create an empty array to store the headlines
  all_headlines = np.empty((0,), dtype=str)

  for date in dates:
      # Set the date range for each day
      from_date = date.isoformat()
      to_date = (date + datetime.timedelta(days=1)).isoformat()

      # Fetch all headlines for the day
      query = f"{company} AND (stock OR market OR finance OR investment OR trading)"
      url = f"https://newsapi.org/v2/everything?q={query}&language=en&apiKey={api_key}&from={from_date}&to={to_date}"
      response = requests.get(url)
      data = response.json()

      print(data["status"])
      print(data["code"])
      print(data["message"])

      if data["status"] == "ok":
          # Append each headline to the array
          headlines = np.array([article["title"] for article in data["articles"][:10]])
          all_headlines = np.concatenate((all_headlines, headlines))
      else:
          print(f"Error fetching headlines for {date}.")

  # Print the final array
  num_headlines = int(len(all_headlines)/10)
  twod_headlines = all_headlines.reshape((num_headlines,10))
  #print(twod_headlines)
  #print(twod_headlines.shape)

  combined = []

  for row in range (0, num_headlines):
      combined.append(' '.join(str(x) for x in twod_headlines[row,:]))

  #print(combined)
  combined = np.array(combined)
  #print(combined.shape)

  compound = []
  SIA = 0

  for i in range (0, len(combined)):
        SIA = getSIA(combined[i])
        compound.append(SIA['compound'])

  print("SCORES +", compound)
  return compound

def getSIA(text):
  sia = SentimentIntensityAnalyzer()
  sentiment = sia.polarity_scores(text)
  return sentiment