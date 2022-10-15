#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 08:08:16 2022

@author: dataguy
"""

# Import libraries
import pandas as pd
import pandas_datareader as pdr
import time
from multiprocessing import Pool, cpu_count
import numpy as np
import itertools
from operator import itemgetter
import matplotlib.pyplot as plt

# Read tickers
table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
tickers = table[0]
symbols = tickers['Symbol'].unique()

# Init
num_cpu = cpu_count()

# Function to get price
def get_ticker_data(symbol):
    
    try:
    
        data = pdr.get_data_yahoo(symbol, '2020-01-01', '2022-10-13').rename({'Adj Close': 'price'}, axis = 1)[['price']]

        # Calculate
        data['ticker'] = symbol
        data['daily_pct_change'] = data['price'] / data['price'].shift(1) - 1
        data['daily_pct_change'].fillna(0, inplace = True)
        data['cum_daily_return'] = (1 + data['daily_pct_change']).cumprod()
        data['short'] = data['price'].rolling(30).mean()
        data['long'] = data['price'].rolling(90).mean()

        return data
    
    except:
    
        return None
    
# Function to set params
def calculate_parameters(df, n1 = None, n2 = None):
    
    if n1 is not None:
        
        df['short'] = df['price'].rolling(n1).mean()
        
    if n2 is not None:
        
        df['long'] = df['price'].rolling(n2).mean()

# Function to perform backtesting
def perform_backtesting(nm):
    
    grp = df2.loc[df2['ticker'] == nm]
    
    positions[nm] = 1000 * grp['signal'] 

    port = positions.multiply(grp['price'], axis = 0)
    pos_diff = positions.diff()

    # Add `holdings` to portfolio
    port['holdings'] = (positions.multiply(grp['price'], axis = 0)).sum(axis = 1)
    
    # Add `cash` to portfolio
    port['cash'] = 10000 - (pos_diff.multiply(grp['price'], axis = 0)).sum(axis = 1).cumsum()   

    # Add `total` to portfolio
    port['total'] = port['cash'] + port['holdings']
    
    # Add `returns` to portfolio
    port['returns'] = port['total'].pct_change() 
    
    port['ticker'] = nm
        
    return port

# Function to perfortm optimization of trading windows
def retrieve_optimized_parameters(symbol):

    score = []
    
    # Execute strategy for optimizing windows
    data = df[df['ticker'] == symbol]
    
    # Backtest windows
    for window in windows:
        
        calculate_parameters(data, window[0], window[1])
        
        # Calculate
        data['signal'] = 0
        data['signal'][window[0]:] = np.where(data['short'][window[0]:] > data['long'][window[0]:], 1.0, 0.0)   
        data['positions'] = data['signal'].diff()
        
        positions = pd.DataFrame(index = data.index).fillna(0.0)
        positions[nm] = 1000 * data['signal'] 

        port = positions.multiply(data['price'], axis = 0)
        pos_diff = positions.diff()

        port['holdings'] = (positions.multiply(data['price'], axis = 0)).sum(axis = 1)
        port['cash'] = 10000 - (pos_diff.multiply(data['price'], axis = 0)).sum(axis = 1).cumsum()   
        port['total'] = port['cash'] + port['holdings']
        
        # Update ticker
        score.append((window, port['total'].iloc[-1]))
        
    # Get max
    best = max(score, key = itemgetter(1))[0] 
    
    return symbol, best

'''
    S&P 500 - Initial
'''

# Iterate tickers to retrieve prices
start = time.time()
pool = Pool(num_cpu - 1)
result = pool.map(get_ticker_data, symbols)  
df = pd.concat([i for i in result])  
end = time.time()
print(f'Elapsed data retrieval time: {round((end - start)/60, 2)} minutes..')

# Calculate trades
df1 = df.dropna()
df2 = pd.DataFrame()

for nm, grp in df1.groupby('ticker'):
    
    try:
        
        grp['signal'] = 0
        grp['signal'][30:] = np.where(grp['short'][30:] > grp['long'][30:], 1.0, 0.0)   
        grp['positions'] = grp['signal'].diff()
    
        df2 = df2.append(grp)
        
    except:
        
        print(f'Error for {nm}..')

# Backtesting
portfolio = pd.DataFrame()

for nm, grp in df2.groupby('ticker'):
    
    positions = pd.DataFrame(index = grp.index).fillna(0.0)
    positions[nm] = 1000 * grp['signal'] 

    port = positions.multiply(grp['price'], axis = 0)
    pos_diff = positions.diff()

    # Add `holdings` to portfolio
    port['holdings'] = (positions.multiply(grp['price'], axis = 0)).sum(axis = 1)

    # Add `cash` to portfolio
    port['cash'] = 10000 - (pos_diff.multiply(grp['price'], axis = 0)).sum(axis = 1).cumsum()   

    # Add `total` to portfolio
    port['total'] = port['cash'] + port['holdings']

    # Add `returns` to portfolio
    port['returns'] = port['total'].pct_change() 
    
    port['ticker'] = nm
    
    portfolio = portfolio.append(port)

# Sum daily returns
agg_portfolio = portfolio.groupby(portfolio.index)['holdings', 'cash', 'total'].agg({'holdings': 'sum',
                                                                                     'cash': 'sum',
                                                                                     'total': 'sum'}, axis = 1)

# Evaluate
print(f"Portfolio starting cash:  ${agg_portfolio['total'].iloc[0]}")
print(f"Portfolio ending cash:    ${round(agg_portfolio['total'].iloc[-1])}")
print(f"Portfolio overall returns: {round((agg_portfolio['total'].iloc[-1]/agg_portfolio['total'].iloc[0]))}X")

'''
    S&P 500 - Optimized
'''

# Create optmization parameters
short_windows = [i for i in range(1, 60, 3)]
long_windows = [i for i in range(90, 180, 3)]
windows = [i for i in itertools.product(*[short_windows, long_windows])]
df3 = pd.DataFrame()

# Iterate tickers to retrieve optimal windows
start = time.time()
pool = Pool(num_cpu - 1)
result = pool.map(retrieve_optimized_parameters, df['ticker'].unique())    
end = time.time()
print(f'Elapsed parameter calculation time: {round((end - start)/60, 2)} minutes..')

# Calculate ledgers based on optimized parameters
for item in result:
    
    data = df[df['ticker'] == item[0]]
    
    data['short'] = data['price'].rolling(item[1][0]).mean()
    data['long'] = data['price'].rolling(item[1][1]).mean()
    
    df3 = df3.append(data)

# Backtest optimized rolling averages 
df4 = pd.DataFrame()
   
for item in result:
    
    grp = df3[df3['ticker'] == item[0]]
    
    try:
        
        grp['signal'] = 0
        grp['signal'][item[1][0]:] = np.where(grp['short'][item[1][0]:] > grp['long'][item[1][0]:], 1.0, 0.0)   
        grp['positions'] = grp['signal'].diff()
    
        df4 = df4.append(grp)
        
    except:
        
        print(f'Error for {nm}..')

# Backtesting
opt_portfolio = pd.DataFrame()

for nm, grp in df4.groupby('ticker'):
    
    positions = pd.DataFrame(index = grp.index).fillna(0.0)
    positions[nm] = 1000 * grp['signal'] 

    port = positions.multiply(grp['price'], axis = 0)
    pos_diff = positions.diff()

    # Add `holdings` to portfolio
    port['holdings'] = (positions.multiply(grp['price'], axis = 0)).sum(axis = 1)

    # Add `cash` to portfolio
    port['cash'] = 10000 - (pos_diff.multiply(grp['price'], axis = 0)).sum(axis = 1).cumsum()   

    # Add `total` to portfolio
    port['total'] = port['cash'] + port['holdings']

    # Add `returns` to portfolio
    port['returns'] = port['total'].pct_change() 
    
    port['ticker'] = nm
    
    opt_portfolio = opt_portfolio.append(port)

# Sum daily returns
agg_opt_portfolio = opt_portfolio.groupby(opt_portfolio.index)['holdings', 'cash', 'total'].agg({'holdings': 'sum',
                                                                                                 'cash': 'sum',
                                                                                                 'total': 'sum'}, axis = 1)

# Evaluate
print(f"Portfolio starting cash:  ${agg_opt_portfolio['total'].iloc[0]}")
print(f"Portfolio ending cash:    ${round(agg_opt_portfolio['total'].iloc[-1])}")
print(f"Portfolio overall returns: {round((agg_opt_portfolio['total'].iloc[-1]/agg_opt_portfolio['total'].iloc[0]))}X")

'''
    Trade Analysis
'''

# Visualize
plt.figure(figsize = (50, 20))
plt.plot(agg_portfolio['total'])
plt.title('Short/Long Trading for S&P 500: 2020 - 2022', fontsize = 24)
plt.xlabel('Date', fontsize = 14)
plt.ylabel('Total Portfolio $ (Million)', fontsize = 14)

# Visualize
plt.figure(figsize = (50, 20))
plt.plot(agg_opt_portfolio['total'])
plt.title('Optimized Short/Long Trading for S&P 500: 2020 - 2022', fontsize = 24)
plt.xlabel('Date', fontsize = 14)
plt.ylabel('Total Portfolio $ (Million)', fontsize = 14)

# Tabulate categories
tickers.groupby(['GICS Sector', 'GICS Sub-Industry']).size().reset_index(name = 'count').sort_values('count', ascending = False)
tickers['category'] = tickers['GICS Sector'] + ' / ' + tickers['GICS Sub-Industry']

# Merge categories
categorical_opt_portfolio = opt_portfolio.reset_index().merge(tickers[['Symbol', 'category']], how = 'left', right_on = 'Symbol', left_on = 'ticker').drop('Symbol', axis = 1)
categorical_opt_portfolio.index = pd.DatetimeIndex(categorical_opt_portfolio['Date'])
categorical_opt_portfolio.drop('Date', axis = 1, inplace = True)
categorical_opt_portfolio = categorical_opt_portfolio.groupby([categorical_opt_portfolio.category, categorical_opt_portfolio.index])['holdings', 'cash', 'total'].agg({'holdings': 'sum',
                                                                                                                                                                       'cash': 'sum',
                                                                                                                                                                       'total': 'sum'}, axis = 1).reset_index()

# Analyze
analysis = pd.DataFrame()

for nm, grp in categorical_opt_portfolio.groupby('category'):
    
    print(f'Showing Results for {nm}: ')
    
    sizing = len(tickers[tickers['category'] == nm])
    start = int(grp['total'].iloc[0])
    finish = int(grp['total'].iloc[-1])
    overall = finish / start
    
    print(f'Size of Sector- {sizing}')
    print(f"Start Investment: ${start}")
    print(f"Ending Balance: ${finish}")
    print(f"Overall Return for Sector- {overall}")
    print()
    
    analysis = analysis.append({'category': nm,
                                'size': sizing,
                                'investment': start,
                                'end_balance': finish,
                                'return': overall}, ignore_index = True)

print('Performance Report by Sector for S&P500: 2020 - 2022')
for idx, row in analysis.sort_values('return', ascending = False).iterrows():
    
    print(f"{row['category']} - ${row['end_balance']} - {round(row['return'], 2)}X")