from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd   # ✅ IMPORTANT (added)
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import os

app = Flask(__name__)

USERNAME = "admin"
PASSWORD = "1234"

# ---------------- LOGIN ----------------
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == USERNAME and password == PASSWORD:
            return redirect(url_for('dashboard'))

    return render_template('login.html')


# ---------------- MANUAL PAGE ----------------
@app.route('/manual')
def manual():
    return render_template('manual.html')


# ---------------- DASHBOARD ----------------
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


# ---------------- ABOUT ----------------
@app.route('/about')
def about():
    return render_template('about.html')


# ---------------- PERFORMANCE ----------------
@app.route('/performance')
def performance():
    return render_template('performance.html')


# ---------------- MANUAL PREDICTION ----------------
@app.route('/manual_predict', methods=['POST'])
def manual_predict():

    stock = request.form['stock']
    open_price = float(request.form['open'])
    high_price = float(request.form['high'])
    low_price = float(request.form['low'])
    close_price = float(request.form['close'])
    volume = float(request.form['volume'])

    model = load_model(f"models/{stock}_model.keras")

    input_data = np.array([[open_price, high_price, low_price, close_price, volume]])

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_input = scaler.fit_transform(input_data)

    X_test = np.repeat(scaled_input, 60, axis=0)
    X_test = np.array([X_test])

    predicted_price = model.predict(X_test)

    dummy = np.zeros((1,5))
    dummy[:,3] = predicted_price[:,0]
    predicted_price = scaler.inverse_transform(dummy)[:,3]

    return render_template('manual.html',
                           prediction=round(float(predicted_price[0]),2),
                           stock=stock)


# ---------------- AUTO PREDICTION ----------------
@app.route('/predict', methods=['POST'])
def predict():

    stock = request.form['stock']

    df = yf.download(stock, start="2026-02-01")
    df = df.dropna()

    dataset = df[['Open','High','Low','Close','Volume']]

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    last_60_days = scaled_data[-60:]
    X_test = np.array([last_60_days])

    model = load_model(f"models/{stock}_model.keras")

    predicted_price = model.predict(X_test)

    dummy = np.zeros((1,5))
    dummy[:,3] = predicted_price[:,0]
    predicted_price = scaler.inverse_transform(dummy)[:,3]

    predicted_price = round(float(predicted_price[0]),2)

    # ✅ ---------------- PREDICTION DATE ----------------
    last_date = df.index[-1]
    next_date = last_date + pd.tseries.offsets.BDay(1)

    # ---------------- GRAPH 1 ----------------
    plt.figure(figsize=(6,3))
    plt.plot(df['Close'].tail(60))
    plt.title(f"{stock} - Last 60 Days Close")
    plt.tight_layout()
    plt.savefig("static/trend.png")
    plt.close()

    # ---------------- GRAPH 2 ----------------
    plt.figure(figsize=(6,3))
    plt.plot(df['Close'].tail(30), label="Recent Prices")
    plt.scatter(29, predicted_price, label="Predicted", marker='o')
    plt.title(f"{stock} Prediction")
    plt.legend()
    plt.tight_layout()
    plt.savefig("static/prediction.png")
    plt.close()

    return render_template('dashboard.html',
                           prediction=predicted_price,
                           stock=stock,
                           prediction_date=next_date.date(),  # ✅ Added
                           show_graph=True)


if __name__ == '__main__':
    app.run(debug=True)