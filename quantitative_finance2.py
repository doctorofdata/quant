#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 08:10:51 2021

@author: operator
"""

# Import libraries
import os
os.chdir('/Users/operator/Documents/')
from quantitative_finance_lib import *
import itertools

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

# Initialize
short_windows = [i for i in range(1, 30, 1)]
long_windows = [i for i in range(30, 60, 1)]
windows = [i for i in itertools.product(*[short_windows, long_windows])]

# Iterate
master = pd.DataFrame()
balance = 0

for ticker in tickers:
    
    stock = stock_dat(ticker, '2000-01-01', '2021-03-02', windows)
    stock.execute_opt()
    
    balance += stock.res['total'].iloc[-1]
    
    print(f'\nOptima calculated for {ticker}: ')
    print(f'Running balance for portfolio =    ${round(balance, 2)}')
    
    master = master.append(stock.res)