#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 06:57:48 2021

@author: operator
"""

import pandas as pd

# Function to execute backtesting
class simulate_investment:
    
    def __init__(self, df, amounts):
        
        self.df = df
        self.amounts = amounts
        self.positions = None
        self.res = None
        
    def build_positions(self):
        
        pos = pd.DataFrame(index = self.df.index)

        # Buy 1000 shares
        for x in self.amounts:
            
            grp = self.df.loc[self.df['ticker'] == x[0]]
            
            pos[x[0]] = 1000 * grp['signal']

            # Calculate trades
            portfolio = pos.multiply(grp['price'], axis = 0)
            pos_diff = pos.diff()
        
        # Update
        self.positions = pos
        
    def calculate_trades(self):
        
        # Cumulative
        portfolio['holdings'] = (positions.multiply(grp['price'], axis = 0)).sum(axis = 1)
        portfolio['cash'] = x[1] - (pos_diff.multiply(grp['price'], axis = 0)).sum(axis = 1).cumsum()   
        portfolio['total'] = portfolio['cash'] + portfolio['holdings']
                
        self.res = portfolio[['holdings', 'cash', 'total']]