import yfinance as yf
import pandas as pd
import os


def main(start_date, end_date):

    #   list of 10 FNGU
    ticker_list = ['TWTR', 'TSLA', 'AAPL', 'FB',
                   'GOOGL', 'BABA', 'NFLX', 'AMZN',
                   'BIDU', 'NVDA']

    for ticker in ticker_list:
        tickerData = yf.Ticker(ticker)
        tickerDf = tickerData.history(period='1d', start=start_date,
                                      end=end_date, auto_adjust=True)

        temp_df = pd.DataFrame(tickerDf).reset_index()
        df = temp_df[['Date', 'Open', 'High', 'Low',
                      'Close', 'Volume', 'Dividends', 'Stock Splits']]
        df.columns = ['date', 'open', 'high', 'low',
                      'close', 'volume', 'dividend', 'split']
        df.to_csv(ticker+".csv")


if __name__ == '__main__':
    path = "C:/Users/16477/Desktop/zipline/dat"
    os.chdir(path)
    main(start_date='2018-1-1', end_date='2021-1-10')

