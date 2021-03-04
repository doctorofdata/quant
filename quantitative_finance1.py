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
from operator import itemgetter

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

# Test lib on sample
ticker = stock_dat('BTC-USD', '2000-01-01', '2021-03-02')
ticker.calculate_parameters(5, 15)
ticker.build_signals()
ticker.execute_backtesting(10000)

# Optimization
short_windows = [i for i in range(1, 30, 1)]
long_windows = [i for i in range(30, 60, 1)]
windows = [i for i in itertools.product(*[short_windows, long_windows])]

scores = []

for window in windows:
    
    ticker.calculate_parameters(window[0], window[1])
    ticker.build_signals()
    ticker.execute_backtesting(10000)
    
    score = ticker.res['total'].iloc[-1]
    
    scores.append((window, score))
    
# Retrieve
best = max(scores, key = itemgetter(1))[0]

# Final calculation
ticker.calculate_parameters(best[0], best[1])
ticker.build_signals()
ticker.execute_backtesting(10000)
