# Crypto Forecasting and Sentiment Analysis App

# Crypto Trend Analysis and Prediction with Prophet

# Problem Definition
### Investors are always in search of investment opportunities where they can grow their wealth and it is essential tto have a well diversified portfolio in order to get the highest level of return for the lowest level of risk. 

### The rationale behind the project is to ensure investors are exposed to wide array of investment alternatives which include cryptocurrencies, to enable investors make informed decision on appropriate portfolio mix and increase their return on investment.

# The project aims to:
### 1. confirm the return on investment (ROI) on a cryptocurrency with a view to determine its profitability by considering the ROI of the stock over a defined period
### 2. establish cryptocurrency price movement by conducting a trend Analysis of cryptocurrency performance and forecasting the future price using the Prophet.
### 3. analyze daily market sentiment of investors.

# Data Preprocessing & Exploratory Data Analysis (EDA)
# Procedure:
### 1. The necessary libraries were imported
### 2. The timeframe of the data to be downloaded from Yahoo Finance was defined and the necessary Ticker symbol was inserted for loading
### 3. The datasets were pre-processed to prepare the data for analysis:
### 4. The datasets were imported from Yahoo Finance and aggregated into a dataframe
### 5. The index was reset, and the necessary data were extracted from the dataframe
### 6. The model was trained to fit the pre-processed data using the Prophet
### 7. The model was evaluated to detect anomalies
### 8. The model was used to predict the price of selected stock using Matplotlib and the Prophet.
### 8. The market sentiment analysis was done using VADER sentiment analysis


# Model Accuracy & Justification
### The accuracy of the model prediction of the cryptocurrency price can be deduced by comparing the line chart showing the trend analysis of historical price movement (actual) of the selected coin vis a viz the trendline of the Prophet forecast. 
### Ideally, Prophet should be able to closely predict the pattern of share price movement after fitting the model on historical price except there are anomalies such as external shocks that could result in either a decline or spike in the price of the coin (which should be further investigated). 

# Deployment Functionality & User Interface
### To ensure a friendly and robust user interface, streamlit was used to deploy the frontend of the model by integrating the app.py model with streamlit url on Github. 

## The app has a simple interactive interface whereby a user will perform the following 3 actions:
### 1.	select the cryptocurrency of your choice and set a required timeframe for the output data
### 2.	select the number of days of forecast you want and
### 3.	run the forecast
### 4.	refresh the sentiment news (to gauge latest pulse of investors on the crypto market)

## In return you will get the following output after clicking on the run the forecast button:
### 1.	Trend Analysis of the selected cryptocurrency over the selected period
### 2.	The Return on Investment (ROI) of the cryptocurrency over the selected time frame
### 3.	Share price information for the last 5 days
### 4.	Option to download the share price data for the selected time range
### 5.	Model Forecast trend chart covering selected timeframe and the near future
### 6.	The Prophet share price trend forecast (with option to zoom in and out over specific time frame)
### 7.	Option to download the share price forecast data for the selected time range

## Crypto App Link : https://crypto-forecasting-sentiment-analysis-app-knvuuoymmybd2e9uwhs9.streamlit.app
# Report & Code Quality
### The code and the data is of high quality, the data is extracted from the website of a reputable and reliable organisation (Yahoo Finance), Cointelegraph, CoinDesk and CryptoSlate.

### Several tests had also been carried out on user interface interactivity before deciding on this final outcome. The codes are also flexible and universal; it gives users a wide array of options and flexibility in terms of being able to select a cryptocurrency of their choice and getting the required output as well as sentiment news analysis of their choice from 3 different choices. 
### In addition, users have the option to download both historical and forecasted share price.
