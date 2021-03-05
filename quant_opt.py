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
from bt_lib import *
import xgboost as xgb
import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

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
           'AMC']

# Define
short_windows = [i for i in range(1, 30, 1)]
long_windows = [i for i in range(30, 60, 1)]
windows = [i for i in itertools.product(*[short_windows, long_windows])]

# Initialize
master = pd.DataFrame()
balance = 0
df = pd.DataFrame()

# Iterate
for ticker in tickers:
    
    stock = stock_dat(ticker, '2000-01-01', '2021-03-02', windows)
    stock.execute_opt()
    
    balance += stock.res['total'].iloc[-1]
    
    print(f'\nOptima calculated for {ticker}: ')
    print(f'Running balance for portfolio =    ${round(balance, 2)}')
    
    # Update
    master = master.append(stock.res)
    
    df = df.append(stock.df)
    
# Coalesce
initial_totals = master.groupby(master.index)[['holdings', 'cash', 'total']].agg({'holdings': 'sum',
                                                                                  'cash': 'sum',
                                                                                  'total': 'sum'})

'''
    Markowitz Optimization
'''    
 
mpt = markowitz_portfolio(data)
weights = mpt.opt_weights

# Initialize
shares = []
cols = [i for i in weights.columns]
total_weight = sum([i for i in weights.iloc[0]])
investment = 10000 * len(weights)

# Iterate
for col in cols:
    
    # Allocate
    allocation = weights[col].iloc[0]
    pct = round(allocation / total_weight, 3)
    
    num_shares = investment * pct
    
    # Update
    shares.append((col, num_shares))
    
# Backtesting
mpt_res = {}

# Get signals
positions = pd.DataFrame(index = df.index.drop_duplicates())

for determination in shares:
    
    positions[determination[0]] = determination[1] * df.loc[df['ticker'] == determination[0]]['signal']
    
pos_diff = positions.diff()

# Get prices
portfolio = pd.DataFrame()

for ticker in tickers:
    
    portfolio[ticker] = positions[ticker].multiply(df.loc[df['ticker'] == ticker]['price'], axis = 0)

    portfolio['holdings'] = (positions.multiply(df.loc[df['ticker'] == ticker]['price'], axis = 0)).sum(axis = 1)

    portfolio['cash'] = investment - (pos_diff.multiply(df.loc[df['ticker'] == ticker]['price'], axis = 0)).sum(axis = 1).cumsum()   

    portfolio['total'] = portfolio['cash'] + portfolio['holdings']

    portfolio['returns'] = portfolio['total'].pct_change()

# Coalesce
mpt_portfolio = portfolio[['holdings', 'cash', 'total', 'returns']]
mpt_portfolio = mpt_portfolio.groupby(mpt_portfolio.index).agg({'holdings': 'sum',
                                                                'cash': 'sum',
                                                                'total': 'sum'})

    