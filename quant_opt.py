#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 13:58:51 2021

@author: operator
"""

# Import libraries
import os
os.chdir('/Users/operator/Documents/')
from quantitative_finance_lib import *
from quant_lib import *
from pt_lib import *
import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import numpy as np

# Initialize list of stocks
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
           'AMC',
           'AAL',
           'LUV']

# Define
short_windows = [i for i in range(1, 30, 1)]
long_windows = [i for i in range(30, 60, 1)]
windows = [i for i in itertools.product(*[short_windows, long_windows])]

# Initialize
ledger = pd.DataFrame()
balance = 0
df = pd.DataFrame()

# Iterate
for ticker in tickers:
    
    stock = stock_dat(ticker, '2000-01-01', '2021-03-02', windows)
    stock.execute_opt(10000)
    
    balance += stock.res['total'].iloc[-1]
    
    # Update
    ledger = ledger.append(stock.res)
    
    df = df.append(stock.df)
    
# Coalesce
initial_totals = ledger.groupby(ledger.index)[['holdings', 'cash', 'total']].agg({'holdings': 'sum',
                                                                                  'cash': 'sum',
                                                                                  'total': 'sum'})

'''
    Markowitz Optimization
'''    
 
stocks = pd.DataFrame()

for ticker in tickers:
    
    stocks[ticker] = df[df['ticker'] == ticker]['price']

logret = np.log(stocks / stocks.shift(1))

np.random.seed(100)
num_ports = 10000
all_weights = np.zeros((num_ports, len(stocks.columns)))
ret_arr = np.zeros(num_ports)
vol_arr = np.zeros(num_ports)
sharpe_arr = np.zeros(num_ports)

for x in range(num_ports):
    
    # Weights
    weights = np.array(np.random.random(len(stocks.columns)))
    weights = weights/np.sum(weights)
    
    # Save weights
    all_weights[x, :] = weights
    
    # Expected return
    ret_arr[x] = np.sum( (logret.mean() * weights * 252))
    
    # Expected volatility
    vol_arr[x] = np.sqrt(np.dot(weights.T, np.dot(logret.cov() * 252, weights)))
    
    # Sharpe Ratio
    sharpe_arr[x] = ret_arr[x] / vol_arr[x]
    
# Get best result
print(f'Location of Best Allocation- {sharpe_arr.argmax()}')

best = []
   
for x, y in zip(stocks.columns, all_weights[sharpe_arr.argmax()]):

    if round(y, 2) != 0:
        
        best.append((x, y))

for b in best:

    print(f'EF Allocation for {b[0]} = {round(b[1], 2)}')    

    