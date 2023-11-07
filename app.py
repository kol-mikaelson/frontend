import os
import shutil
from flask import Flask,render_template

app = Flask(__name__)
import matplotlib
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
import yfinance as yf
from pandas_datareader import data as pdr
from keras.models import Sequential
from keras.layers import Dense, LSTM
yf.pdr_override()


from datetime import datetime

tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN','META']
end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)

for stock in tech_list:
    globals()[stock] = yf.download(stock, start, end)

company_list = [AAPL, GOOG, MSFT, AMZN, META]
company_name = ["APPLE", "GOOGLE", "MICROSOFT", "AMAZON","META"]

for company, com_name in zip(company_list, company_name):
    company["company_name"] = com_name

df = pd.concat(company_list, axis=0)
df.tail(10)
plt.figure(figsize=(15, 10))
plt.subplots_adjust(top=1.25, bottom=1.2)



def closeprice():
    for i, company in enumerate(company_list, 1):
        plt.subplot(3, 3, i)
        company['Adj Close'].plot()
        plt.ylabel('Adj Close')
        plt.xlabel(None)
        plt.title(f"Closing Price of {tech_list[i - 1]}")

    print(plt.tight_layout())

    plt.savefig("static\\close")
    plt.figure(figsize=(15, 10))
    plt.subplots_adjust(top=1.25, bottom=1.2)

def vol():
    for i, company in enumerate(company_list, 1):
        plt.subplot(3, 3, i)
        company['Volume'].plot()
        plt.ylabel('Volume')
        plt.xlabel(None)
        plt.title(f"Sales Volume for {tech_list[i - 1]}")
    plt.autoscale()
    plt.tight_layout()
    plt.savefig("static\\volume1")
    ma_day = [10, 20, 50]

    for ma in ma_day:
        for company in company_list:
            column_name = f"MA for {ma} days"
            company[column_name] = company['Adj Close'].rolling(ma).mean()

    fig, axes = plt.subplots(nrows=2, ncols=2)
    fig.set_figheight(10)
    fig.set_figwidth(15)

    AAPL[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[0, 0])
    axes[0, 0].set_title('APPLE')

    GOOG[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[0, 1])
    axes[0, 1].set_title('GOOGLE')

    MSFT[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[1, 0])
    axes[1, 0].set_title('MICROSOFT')

    AMZN[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[1, 1])
    axes[1, 1].set_title('AMAZON')
    META[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[1, 1])
    axes[1, 1].set_title('META')
    fig.tight_layout()
    fig.savefig("static\\volume2")
def dayreturn():
    for company in company_list:
        company['Daily Return'] = company['Adj Close'].pct_change()
    fig, axes = plt.subplots(nrows=2, ncols=2)
    fig.set_figheight(10)
    fig.set_figwidth(15)

    AAPL['Daily Return'].plot(ax=axes[0,0], legend=True, linestyle='--', marker='o')
    axes[0,0].set_title('APPLE')

    GOOG['Daily Return'].plot(ax=axes[0,1], legend=True, linestyle='--', marker='o')
    axes[0,1].set_title('GOOGLE')

    MSFT['Daily Return'].plot(ax=axes[1,0], legend=True, linestyle='--', marker='o')
    axes[1,0].set_title('MICROSOFT')

    AMZN['Daily Return'].plot(ax=axes[1,1], legend=True, linestyle='--', marker='o')
    axes[1,1].set_title('AMAZON')

    fig.tight_layout()
    fig.savefig("static\\day1")

    plt.figure(figsize=(12, 9))

    for i, company in enumerate(company_list, 1):
        plt.subplot(3, 3, i)
        company['Daily Return'].hist(bins=50)
        plt.xlabel('Daily Return')
        plt.ylabel('Counts')
        plt.title(f'{company_name[i - 1]}')

    plt.tight_layout()
    plt.savefig("static\\dayreturn")

def corelation():
    closing_df = pdr.get_data_yahoo(tech_list, start=start, end=end)['Adj Close']
    tech_rets = closing_df.pct_change()
    tech_rets.head()

    sns.jointplot(x='GOOG', y='GOOG', data=tech_rets, kind='scatter', color='seagreen')


    sns.jointplot(x='GOOG', y='MSFT', data=tech_rets, kind='scatter')


    sns.pairplot(tech_rets, kind='reg')

    return_fig = sns.PairGrid(tech_rets.dropna())

    return_fig.map_upper(plt.scatter, color='purple')

    return_fig.map_lower(sns.kdeplot, cmap='cool_d')

    return_fig.map_diag(plt.hist, bins=30)

    returns_fig = sns.PairGrid(closing_df)

    returns_fig.map_upper(plt.scatter,color='purple')

    returns_fig.map_lower(sns.kdeplot,cmap='cool_d')

    returns_fig.map_diag(plt.hist,bins=30)

    plt.figure(figsize=(12, 10))

    plt.subplot(3, 3, 1)
    sns.heatmap(tech_rets.corr(), annot=True, cmap='summer')
    plt.title('Correlation of stock return')
    plt.savefig("static\\corelation1")

    plt.subplot(3, 3, 2)
    sns.heatmap(closing_df.corr(), annot=True, cmap='summer')
    plt.title('Correlation of stock closing price')
    plt.savefig("static\\corelation2")

def predict():
    df = pdr.get_data_yahoo('AAPL', start='2012-01-01', end=datetime.now())
    plt.figure(figsize=(16, 6))
    plt.title('Close Price History')
    plt.plot(df['Close'])
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.show()
    data = df.filter(['Close'])
    dataset = data.values
    training_data_len = int(np.ceil(len(dataset) * .95))

    training_data_len
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    scaled_data
    train_data = scaled_data[0:int(training_data_len), :]
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])
        if i <= 61:
            print(x_train)
            print(y_train)
            print()
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1)
    test_data = scaled_data[training_data_len - 60:, :]
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])
    x_test = np.array(x_test)

    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
    rmse
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    plt.figure(figsize=(16, 6))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.savefig("static\\prediction")
def rti():
    closing_df = pdr.get_data_yahoo(tech_list, start=start, end=end)['Adj Close']

    # Make a new tech returns DataFrame
    tech_rets = closing_df.pct_change()
    rets = tech_rets.dropna()

    area = np.pi * 20

    plt.figure(figsize=(10, 8))
    plt.scatter(rets.mean(), rets.std(), s=area)
    plt.xlabel('Expected return')
    plt.ylabel('Risk')

    for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
        plt.annotate(label, xy=(x, y), xytext=(50, 50), textcoords='offset points', ha='right', va='bottom',
                     arrowprops=dict(arrowstyle='-', color='blue', connectionstyle='arc3,rad=-0.3'))
    plt.tight_layout()
    plt.savefig("static\\rti")
@app.route('/')
def hello_world():  # put application's code here
    return render_template("Untitled-1.html")

@app.route('/close')
def cp():
    closeprice()
    return render_template("close.html")
@app.route('/vos')
def vosp():
    vol()
    return render_template("vos.html")
@app.route('/dr')
def drp():
    dayreturn()
    return render_template("dr.html")
@app.route('/rti')
def rtip():
    rti()
    return render_template("rti.html")
@app.route('/scr')
def scrp():
    corelation()
    return render_template("scr.html")
@app.route('/smp')
def smpp():
    predict()
    return render_template("smp.html")
if __name__ == '__main__':
    app.run()
