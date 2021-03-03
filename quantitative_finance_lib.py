#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 12:00:40 2021

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
from scipy.optimize import brute

# Class for computing individual stock
class stock_dat:
    
    def __init__(self, ticker, start, end):
        
        self.ticker = ticker
        self.start = start
        self.end = end
        self.df = self.grab_dat()
        self.res = None
            
    # Function to retrieve data
    def grab_dat(self):
        
        dat = pdr.get_data_yahoo(self.ticker, self.start, self.end).rename({'Adj Close': 'price'}, axis = 1)[['price']]
        
        dat['ticker'] = self.ticker

        # Returns
        dat['daily_pct_change'] = dat['price'] / dat['price'].shift(1) - 1
        dat['daily_pct_change'].fillna(0, inplace = True)
        dat['cum_daily_return'] = (1 + dat['daily_pct_change']).cumprod()
        
        return dat
        
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
        
        self.res = portfolio 
        
        return portfolio['total'].iloc[-1]
    

# Execute
if __name__ == '__main__':
    
    ticker = stock_dat('BTC-USD', '2000-01-01', '2021-03-02')
    ticker.calculate_parameters(5, 15)
    ticker.build_signals()
    ticker.execute_backtesting(10000)