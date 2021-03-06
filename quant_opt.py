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
ledger = pd.DataFrame()
balance = 0
df = pd.DataFrame()

# Iterate
for ticker in tickers:
    
    stock = stock_dat(ticker, '2000-01-01', '2021-03-02', windows)
    stock.execute_opt(10000)
    
    balance += stock.res['total'].iloc[-1]
    
    print(f'\nOptima calculated for {ticker}: ')
    print(f'Running balance for portfolio =    ${round(balance, 2)}')
    
    # Update
    ledger = ledger.append(stock.res)
    
    df = df.append(stock.df)
    
# Coalesce
initial_totals = master.groupby(master.index)[['holdings', 'cash', 'total']].agg({'holdings': 'sum',
                                                                                  'cash': 'sum',
                                                                                  'total': 'sum'})

'''
    Markowitz Optimization
'''    
 
mpt = markowitz_portfolio(df)
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

# Perform adjusted backtesting
mpt_ledger = pd.DataFrame()
mpt_balance = 0

for info in shares:
    
    stock = stock_dat(info[0], '2000-01-01', '2021-03-02', windows)
    stock.execute_opt(info[1])
    
    mpt_balance += stock.res['total'].iloc[-1]
    
    print(f'\nOptima calculated for {info[0]}: ')
    print(f'Running balance for portfolio =    ${round(mpt_balance, 2)}')
    
    # Update
    mpt_ledger = mpt_ledger.append(stock.res)
    
# Coalesce
mpt_totals = mpt_ledger.groupby(mpt_ledger.index)[['holdings', 'cash', 'total']].agg({'holdings': 'sum',
                                                                                      'cash': 'sum',
                                                                                      'total': 'sum'})
        
# Visualize
fig, ax = plt.subplots(figsize = (10, 6))
plt.title('Cumulative Returns for Portfolios', fontsize = 12)
plt.xlabel('Date', fontsize = 12)
plt.ylabel('Cumulative Portfolio Value ($)', fontsize = 12)
ax.plot(initial_totals['total'], lw = .5, label = 'Equally-Weighted')
ax.plot(mpt_totals['total'], lw = .5, label = 'Markowitz')
fig.legend(loc = 'best', fontsize = 12)