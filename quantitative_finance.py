#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 08:10:51 2021

@author: operator
"""

# Import libraries
import os
import pandas_datareader as pdr
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Get
tickers = ['AG', 
           'BABA', 
           'CSTL', 
           'HLT', 
           'IEC', 
           'PYPL', 
           'PINS', 
           'UPLD', 
           'W', 
           'MSFT', 
           'SYK', 
           'SCCO', 
           'AAPL', 
           'GOOGL', 
           'IBM', 
           'USD', 
           'GLD', 
           'TMUS', 
           'T', 
           'CHTR', 
           'CBRE', 
           'AMZN', 
           'NFLX', 
           'TSLA',
           'PGR',
           'LULU',
           'PFE',
           'DLR',
           'TXN',
           'HPE',
           'WBA',
           'MCFE',
           'JPM',
           'CCL',
           'RCL',
           'JWN',
           'CNK',
           'AMC']

data = pd.DataFrame()

for ticker in tickers:
    
     df = pdr.get_data_yahoo(ticker, '2000-01-01', '2021-03-02').rename({'Adj Close': 'price'}, axis = 1)[['price']]
     df['ticker'] = ticker
     
     # Daily returns
     df['daily_pct_change'] = df['price'] / df['price'].shift(1) - 1
     df['daily_pct_change'].fillna(0, inplace = True)
     df['cum_daily_return'] = (1 + df['daily_pct_change']).cumprod()
    
    # Trading
     df['signal'] = 0.0
     df['short'] = df['price'].rolling(window = 5, min_periods = 1, center = False).mean()
     df['long'] = df['price'].rolling(window = 15, min_periods = 1, center = False).mean()
     df['signal'][5:] = np.where(df['short'][5:] > df['long'][5:], 1.0, 0.0)   
     df['positions'] = df['signal'].diff()
     
     data = data.append(df)
    
# Backtesting
initial_capital= float(10000.0)

data1 = pd.DataFrame()

# Create a DataFrame `positions`
positions = pd.DataFrame(index = df.index).fillna(0.0)

for nm, grp in data.groupby('ticker'):
    
    # Buy shares
    positions[nm] = 1000 * grp['signal']   
  
    # Initialize 
    portfolio = positions.multiply(grp['price'], axis = 0)

    # Store the difference in shares owned 
    pos_diff = positions.diff()

    # Add `holdings` to portfolio
    portfolio['holdings'] = (positions.multiply(grp['price'], axis = 0)).sum(axis = 1)

    # Add `cash` to portfolio
    portfolio['cash'] = initial_capital - (pos_diff.multiply(grp['price'], axis = 0)).sum(axis = 1).cumsum()   

    # Add `total` to portfolio
    portfolio['total'] = portfolio['cash'] + portfolio['holdings']

    # Add `returns` to portfolio
    portfolio['returns'] = portfolio['total'].pct_change()

    portfolio['ticker'] = nm
    
    data1 = data1.append(portfolio)

# Coalesce
data2 = data1[['holdings', 'cash', 'total']].groupby(data1.index).agg({'holdings': 'sum',
                                                                    'cash': 'sum',
                                                                    'total': 'sum'}, axis = 1)

# Visualize
fig, ax = plt.subplots(figsize = (10, 6))
ax.plot(data2.loc[data2['holdings'] != 0]['total'], lw = 1)
plt.xlabel('Year')
plt.ylabel('Portfolio Value ($)')
plt.title('Cumulative Returns for Portfolio: 2000 - 2020')