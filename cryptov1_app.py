import streamlit as st
import yfinance as yf
import datetime as dt
from prophet import Prophet
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import pandas as pd
import io
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import seaborn as sns

# Streamlit page config
st.set_page_config(page_title="Crypto Forecast & News Sentiment App", layout="wide")

# Initialize session states
if "run_forecast" not in st.session_state:
    st.session_state["run_forecast"] = False
if "refresh_news" not in st.session_state:
    st.session_state["refresh_news"] = False

# --- Prophet & Price Functions ---
def download_crypto_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        st.error(f"No data found for ticker '{ticker}'. Make sure it's a valid crypto ticker like 'BTC-USD'.")
        return None
    data.reset_index(inplace=True)
    return data

def prepare_prophet_data(data):
    df = data[['Date', 'Close']].copy()
    df.columns = ['ds', 'y']
    return df

def forecast_crypto(data, forecast_days=91):
    model = Prophet(changepoint_prior_scale=0.15, yearly_seasonality=True, daily_seasonality=True)
    model.fit(data)
    future = model.make_future_dataframe(periods=forecast_days, freq='D')
    forecast = model.predict(future)
    return model, forecast

def plot_price_trend(data, ticker):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(data['Date'], data['Close'], color='blue', label='Price')
    ax.set_title(f'{ticker} Price Trend')
    ax.set_xlabel('Date')
    ax.set_ylabel(f'Close Price (USD)')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button("📄 Download Price Trend CSV", csv, f"{ticker}_price_trend.csv", "text/csv")

    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png')
    st.download_button("🖼️ Download Price Trend Chart", img_buf.getvalue(), f"{ticker}_price_trend.png", "image/png")

def plot_forecast(model, forecast, ticker):
    fig = model.plot(forecast, xlabel='Date', ylabel=f'{ticker} Price (USD)')
    plt.title(f'{ticker} Price Forecast')
    st.pyplot(fig)

def plot_prediction_variance(data, forecast, ticker):
    fig1, ax1 = plt.subplots(figsize=(18, 7))
    ax1.plot(data['ds'], data['y'], label="Actual", linewidth=2, color='black')
    ax1.plot(forecast['ds'], forecast['yhat_lower'], label="Predicted lower", linewidth=2, color='green')
    ax1.plot(forecast['ds'], forecast['yhat_upper'], label="Predicted upper", linewidth=2, color='red')
    ax1.set_title(f'{ticker} Price with Prediction Variance')
    ax1.set_xlabel('Date')
    ax1.set_ylabel(f'{ticker} Price (USD)')
    ax1.legend()
    ax1.grid(True)
    st.pyplot(fig1)

    trace_actual = go.Scatter(name='Actual price', mode='markers',
        x=list(data['ds']), y=list(data['y']),
        marker=dict(color='black', line=dict(width=2)))
    trace_trend = go.Scatter(name='Trend', mode='lines',
        x=list(forecast['ds']), y=list(forecast['yhat']),
        line=dict(color='red', width=3))
    trace_upper = go.Scatter(name='Upper band', mode='lines',
        x=list(forecast['ds']), y=list(forecast['yhat_upper']),
        line=dict(color='#57b88f'), fill='tonexty')
    trace_lower = go.Scatter(name='Lower band', mode='lines',
        x=list(forecast['ds']), y=list(forecast['yhat_lower']),
        line=dict(color='#1705ff'))

    fig_plotly = go.Figure(data=[trace_actual, trace_trend, trace_lower, trace_upper])
    fig_plotly.update_layout(title=f'{ticker} Forecast using Prophet (Plotly)',
                             xaxis_title='Date', yaxis_title=f'{ticker} Price (USD)')
    st.plotly_chart(fig_plotly, use_container_width=True)

def calculate_roi(data, ticker):
    start_price = float(data['Close'].iloc[0])
    end_price = float(data['Close'].iloc[-1])
    roi = ((end_price - start_price) / start_price) * 100
    st.markdown(f"### ROI for {ticker}: `{roi:.2f}%` from {data['Date'].iloc[0].date()} to {data['Date'].iloc[-1].date()}`")

# --- Sentiment Analysis & RSS ---
analyzer = SentimentIntensityAnalyzer()

RSS_FEEDS = {
    "CoinTelegraph": "https://cointelegraph.com/rss",
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "CryptoSlate": "https://cryptoslate.com/feed/"
}

@st.cache_data(ttl=3600)
def fetch_and_analyze_news(rss_url, max_articles=10):
    feed = feedparser.parse(rss_url)
    news_items = []
    for entry in feed.entries[:max_articles]:
        title = entry.title
        published = entry.published if 'published' in entry else "N/A"
        sentiment = analyzer.polarity_scores(title)['compound']
        news_items.append({
            "title": title,
            "sentiment": sentiment,
            "link": entry.link,
            "published": published
        })
    return news_items

def plot_sentiment_distribution(news):
    sentiments = [item["sentiment"] for item in news]
    sentiment_labels = ["Positive" if s > 0.1 else "Negative" if s < -0.1 else "Neutral" for s in sentiments]
    df_sentiments = pd.DataFrame(sentiment_labels, columns=["Sentiment"])

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df_sentiments, x="Sentiment",
                  palette={"Positive": "green", "Negative": "red", "Neutral": "gray"}, ax=ax)
    ax.set_title("News Sentiment Distribution")
    st.pyplot(fig)

# --- Streamlit UI ---
st.title("💰 Cryptocurrency Forecast & News Sentiment App")

# Sidebar inputs
st.sidebar.header("User Inputs")
crypto_ticker = st.sidebar.text_input("Enter Crypto Ticker (e.g. BTC-USD, ETH-USD)", "BTC-USD")
start_date = st.sidebar.date_input("Start Date", dt.date(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", dt.date.today())
forecast_days = st.sidebar.slider("Forecast Days", min_value=30, max_value=365, value=90, step=30)

# Forecast trigger
if st.sidebar.button("Run Forecast"):
    st.session_state["run_forecast"] = True

# News sentiment settings
st.sidebar.header("News & Sentiment Settings")
selected_feed = st.sidebar.selectbox("Select News Source", list(RSS_FEEDS.keys()))

# News refresh trigger
if st.sidebar.button("🔄 Refresh News"):
    st.session_state["refresh_news"] = True

# Optional reset
if st.sidebar.button("🔄 Reset Page"):
    st.session_state["run_forecast"] = False
    st.session_state["refresh_news"] = False

# Forecast display
if st.session_state["run_forecast"]:
    data = download_crypto_data(crypto_ticker, start_date, end_date)
    if data is not None:
        st.subheader(f"Raw data for {crypto_ticker}")
        st.dataframe(data.tail())

        st.subheader("Price Trend")
        plot_price_trend(data, crypto_ticker)

        calculate_roi(data, crypto_ticker)

        df_prophet_data = prepare_prophet_data(data)

        st.subheader("Forecast Plot (Prophet)")
        model, forecast = forecast_crypto(df_prophet_data, forecast_days)
        plot_forecast(model, forecast, crypto_ticker)

        st.subheader("Prediction Variance")
        plot_prediction_variance(df_prophet_data, forecast, crypto_ticker)

        st.subheader("Forecast Data")
        forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_days)
        st.dataframe(forecast_display)

        forecast_csv = forecast_display.to_csv(index=False).encode('utf-8')
        st.download_button("📄 Download Forecast CSV", forecast_csv, f"{crypto_ticker}_forecast.csv", "text/csv")

# News sentiment display
if st.session_state["refresh_news"]:
    news = fetch_and_analyze_news(RSS_FEEDS[selected_feed])
    st.session_state["refresh_news"] = False

    st.subheader(f"📰 Latest News from {selected_feed} with Sentiment Analysis")

    if news:
        for item in news:
            sentiment_label = "Positive" if item["sentiment"] > 0.1 else "Negative" if item["sentiment"] < -0.1 else "Neutral"
            sentiment_color = "green" if sentiment_label == "Positive" else "red" if sentiment_label == "Negative" else "gray"

            st.markdown(f"**[{item['title']}]({item['link']})**")
            st.markdown(f"<small><i>Published: {item['published']}</i></small>", unsafe_allow_html=True)
            st.markdown(f"<span style='color:{sentiment_color}'>Sentiment: {sentiment_label} ({item['sentiment']:.2f})</span>", unsafe_allow_html=True)
            st.markdown("---")

        plot_sentiment_distribution(news)
    else:
        st.info("No news articles found.")
