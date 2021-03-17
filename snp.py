#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 05:50:58 2021

@author: operator
"""

# Import
import os
os.chdir('/Users/operator/Documents/')
from quantitative_finance_lib import *
import pandas as pd
from tqdm import tqdm

# Get tickers
table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
tickers = table[0]

tickers.groupby(['GICS Sector', 'GICS Sub-Industry']).size()
tickers['GICS Sector'].value_counts()

tickers1 = {}

for nm, grp in tickers.groupby('GICS Sector'):
    
    tickers1[nm] = [i for i in grp['Symbol'].unique()]

# Get data    
data = {}

for k, v in tqdm(tickers1.items()):
    
    df = pd.DataFrame()
    
    for ticker in v:
        
        try:
            
            dat = grab_dat(ticker, '2000-01-01', '2021-03-17')
            df = df.append(dat)
            
        except:
            
            pass
    
    df['sector'] = k
    data[k] = df
    
# Coalesce
master = pd.DataFrame()

for k, v in data.items():
    
    master = master.append(v)
    
# Calculate
m1 = pd.DataFrame()

for nm, grp in master.groupby('ticker'):
    
    # Averages
    grp['short'] = grp['price'].rolling(30).mean()
    grp['long'] = grp['price'].rolling(90).mean()

    # Signals
    grp['signal'] = 0
    grp['signal'][30:] = np.where(grp['short'][30:] > grp['long'][30:], 1.0, 0.0)   
    grp['positions'] = grp['signal'].diff()
    
    m1 = m1.append(grp)

# Backtesting
m2 = m1.dropna()
master_portfolio = pd.DataFrame()

for nm, grp in tqdm(m2.groupby('ticker')):

    positions = pd.DataFrame(index = grp.index)
    positions[nm] = 1000 * grp['signal']
    
    portfolio = positions.multiply(grp['price'], axis = 0)
    pos_diff = positions.diff()
        
    portfolio['holdings'] = (positions.multiply(grp['price'], axis = 0)).sum(axis = 1)
    portfolio['cash'] = 10000 - (pos_diff.multiply(grp['price'], axis = 0)).sum(axis = 1).cumsum()   
    portfolio['total'] = portfolio['cash'] + portfolio['holdings']
    portfolio['ticker'] = nm

    master_portfolio = master_portfolio.append(portfolio)
    
# Coalesce
aggregate = master_portfolio.groupby(master_portfolio.index).agg({'holdings': 'sum',
                                                                  'cash': 'sum',
                                                                  'total': 'sum'})

fig, ax = plt.subplots(figsize = (10, 6))
ax.plot(aggregate['total'], lw = .5)
plt.title('Cumulative Returns for Short/Long Trading, S&P500: 2000 - 2021', fontsize = 12)
plt.xlabel('Date', fontsize = 12)
plt.ylabel('Total ($)', fontsize = 12)
#plt.ticklabel_format(axis = "x", style = "sci", scilimits = (0,0))

    
    
    
    
    
    
    