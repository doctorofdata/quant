#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 12:26:41 2021

@author: operator
"""

# Import
import pandas_datareader as pdr
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import itertools
from operator import itemgetter

# Class to retrieve data
class portfolio_data:
    
    def __init__(self, tickers):
        
        self.tickers = tickers
        self.df = None
        
        self.grab_dat()
        
    # Function to retrieve data
    def grab_dat(self):
        
        out = pd.DataFrame()
        
        for ticker in self.tickers:
            
            dat = pdr.get_data_yahoo(ticker).rename({'Adj Close': 'price'}, axis = 1)[['price']]
            dat['ticker'] = ticker

            # Returns
            dat['daily_pct_change'] = dat['price'] / dat['price'].shift(1) - 1
            dat['daily_pct_change'].fillna(0, inplace = True)
            dat['cum_daily_return'] = (1 + dat['daily_pct_change']).cumprod()
            
            out = out.append(dat)
            
        self.df = out
        
# Class for computing individual stock
class strategy_execution:
    
    def __init__(self):
        
        self.ticker = ticker
        self.start = start
        self.end = end
        self.windows = windows
        self.df = self.grab_dat()
        self.res = None
        self.opt = None
        self.calculate_opt()
        
    # Function to set params
    def calculate_parameters(self, n1 = None, n2 = None):
        
        if n1 is not None:
            
            self.n1 = n1
            
            self.df['short'] = self.df['price'].rolling(self.n1).mean()
            
        if n2 is not None:
            
            self.n2 = n2
            
            self.df['long'] = self.df['price'].rolling(self.n2).mean()

    # Function to build signals
    def build_signals(self):
    
        # Signals
        self.df['signal'] = 0
        self.df['signal'][self.n1:] = np.where(self.df['short'][self.n1:] > self.df['long'][self.n1:], 1.0, 0.0)   
        self.df['positions'] = self.df['signal'].diff()
        
    # Function to backtest
    def execute_backtesting(self, initial_capital):
        
        data = self.df.copy().dropna()
                
        positions = pd.DataFrame(index = data.index)
        positions[self.ticker] = 1000 * data['signal']
        
        portfolio = positions.multiply(data['price'], axis = 0)
        pos_diff = positions.diff()
        
        portfolio['holdings'] = (positions.multiply(data['price'], axis = 0)).sum(axis = 1)
        portfolio['cash'] = initial_capital - (pos_diff.multiply(data['price'], axis = 0)).sum(axis = 1).cumsum()   
        portfolio['total'] = portfolio['cash'] + portfolio['holdings']
                
        return portfolio['total'].iloc[-1]
    
    # Function to backtest optima
    def execute_optimal_backtesting(self, initial_capital):

        data = self.df.copy().dropna()
                
        positions = pd.DataFrame(index = data.index)
        positions[self.ticker] = 1000 * data['signal']
        
        portfolio = positions.multiply(data['price'], axis = 0)
        pos_diff = positions.diff()
        
        portfolio['holdings'] = (positions.multiply(data['price'], axis = 0)).sum(axis = 1)
        portfolio['cash'] = initial_capital - (pos_diff.multiply(data['price'], axis = 0)).sum(axis = 1).cumsum()   
        portfolio['total'] = portfolio['cash'] + portfolio['holdings']
                
        self.res = portfolio
    
    # Function to optimize parameters
    def calculate_opt(self):
        
        scores = []

        for window in self.windows:
    
            self.calculate_parameters(window[0], window[1])
            self.build_signals()
            self.execute_backtesting(10000)
    
            score = self.execute_backtesting(10000)
    
            scores.append((window, score))
    
        # Retrieve
        self.opt = max(scores, key = itemgetter(1))[0]
    
    # Function to build optima
    def execute_opt(self):
        
        self.calculate_parameters(self.opt[0], self.opt[1])
        self.build_signals()
        self.execute_optimal_backtesting(10000)

