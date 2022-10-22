#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 08:08:16 2022

@author: dataguy
"""

# Import libraries
import tqdm
import os
os.chdir('/home/dataguy/Documents/')
import pandas as pd
import pandas_datareader as pdr
import time
from multiprocessing import Pool, cpu_count
import numpy as np
import itertools
from operator import itemgetter
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Init
num_cpu = cpu_count()

# Read tickers
table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
tickers = table[0]
symbols = tickers['Symbol'].unique()

# Function to get price
def get_ticker_data(symbol):
    
    try:
    
        data = pdr.get_data_yahoo(symbol, '2000-01-01', '2022-10-13').rename({'Adj Close': 'price'}, axis = 1)[['price']]

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

# Function to compute annual performance
def compute_portfolio_annual_performance(weights, returns, cov_mat):
    
    return np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights))) * np.sqrt(250), np.sum(returns * weights) * 250

# Function to compute returns
def random_portfolii(num_assets, num, returns, cov_mat, rate):
    
    result = np.zeros((3, num))
    weights_rec = []
    
    for n in range(num):
        
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weights_rec.append(weights)
        
        pstd, pret = compute_portfolio_annual_performance(weights, returns, cov_mat)
        
        result[0, n] = pstd
        result[1, n] = pret
        result[2, n] = (pret - rate) / pstd
        
    return result, weights_rec

# Function to backtest
def get_backtesting_results(symbol):
    
    grp = df1[df1['ticker'] == symbol]
    
    # Calculation
    positions = pd.DataFrame(index = grp.index).fillna(0.0)
    positions[symbol] = 1000 * grp['signal'] 

    port = positions.multiply(grp['price'], axis = 0)
    pos_diff = positions.diff()

    # Computation
    port['holdings'] = (positions.multiply(grp['price'], axis = 0)).sum(axis = 1)
    port['cash'] = 10000 - (pos_diff.multiply(grp['price'], axis = 0)).sum(axis = 1).cumsum()   
    port['total'] = port['cash'] + port['holdings']
    
    return port

# Function to compute portfolio report
def display_simulated_allocation_report(returns, cov_mat, num, rate):
    
    results, weights = random_portfolii(len(returns), num, returns, cov_mat, rate)
    
    max_sharpe_idx = np.argmax(results[2])
    sdp, rp = results[0, max_sharpe_idx], results[1, max_sharpe_idx]
    max_sharpe_allocation = pd.DataFrame(weights[max_sharpe_idx], index = stocks.columns, columns = ['allocation'])
    max_sharpe_allocation.allocation = [round(i * 100,2)for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T
    
    min_vol_idx = np.argmin(results[0])
    sdp_min, rp_min = results[0,min_vol_idx], results[1,min_vol_idx]
    min_vol_allocation = pd.DataFrame(weights[min_vol_idx], index = stocks.columns,columns = ['allocation'])
    min_vol_allocation.allocation = [round(i * 100,2)for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T
    
    print("-"*80)
    print("Maximum Sharpe Ratio Portfolio Allocation\n")
    print("Annualised Return:", round(rp,2))
    print("Annualised Volatility:", round(sdp,2))
    print()
    print(max_sharpe_allocation)
    print("-"*80)
    print("Minimum Volatility Portfolio Allocation\n")
    print("Annualised Return:", round(rp_min,2))
    print("Annualised Volatility:", round(sdp_min,2))
    print()
    print(min_vol_allocation)
    
    plt.figure(figsize = (10, 7))
    plt.scatter(results[0,:],results[1,:],c=results[2,:], cmap = 'YlGnBu', marker = 'o', s = 10, alpha = 0.3)
    plt.colorbar()
    plt.scatter(sdp,rp,marker = '*', color = 'r', s = 500, label = 'Maximum Sharpe ratio')
    plt.scatter(sdp_min, rp_min, marker = '*', color = 'g', s = 500, label = 'Minimum volatility')
    plt.title('Simulated Portfolio Optimization based on Efficient Frontier')
    plt.xlabel('annualised volatility')
    plt.ylabel('annualised returns')
    plt.legend(labelspacing = 0.8)
    
    return max_sharpe_allocation

# Function to compute investment sizing
def optimized_backtesting(allocation):

    grp = df1[df1['ticker'] == allocation[0]]
    
    positions = pd.DataFrame(index = grp.index).fillna(0.0)
    positions[allocation[0]] = (num_shares * allocation[1]) * grp['signal'] 

    port = positions.multiply(grp['price'], axis = 0)
    pos_diff = positions.diff()

    # Add `holdings` to portfolio
    port['holdings'] = (positions.multiply(grp['price'], axis = 0)).sum(axis = 1)

    # Add `cash` to portfolio
    port['cash'] = (total_cash * allocation[1]) - (pos_diff.multiply(grp['price'], axis = 0)).sum(axis = 1).cumsum()   

    # Add `total` to portfolio
    port['total'] = port['cash'] + port['holdings']

    # Add `returns` to portfolio
    port['returns'] = port['total'].pct_change() 
    
    port['ticker'] = allocation[0]
    
    return port.drop(allocation[0], axis = 1)

# Iterate tickers to retrieve prices
start = time.time()
pool = Pool(num_cpu - 1)
result = pool.map(get_ticker_data, symbols)  
df = pd.concat([i for i in result])  
end = time.time()
print(f'Elapsed data retrieval time: {round((end - start)/60, 2)} minutes..')
    
# Calculate trades
df1 = pd.DataFrame()

for nm, grp in df.groupby('ticker'):
    
    try:
        
        grp['signal'] = 0
        grp['signal'][30:] = np.where(grp['short'][30:] > grp['long'][30:], 1.0, 0.0)   
        grp['positions'] = grp['signal'].diff()
    
        df1 = df1.append(grp)
        
    except:
        
        print(f'Error for {nm}..')

# Iterate tickers to coduct backtest
start = time.time()
pool = Pool(num_cpu - 1)
backtest = pool.map(get_backtesting_results, symbols)
end = time.time()
print(f'Elapsed backtesting time: {round((end - start), 2)} seconds..')

# Combine result from each stock
totals = pd.DataFrame()
signals = pd.DataFrame()
backtesting = pd.DataFrame()

for x, y in zip(symbols, backtest):
    
    signals[x] = y[x] 
    
    z = y.drop(x, axis = 1)
    z['ticker'] = x
    
    backtesting = backtesting.append(z)
    
    totals = totals.add(y[['holdings', 'cash', 'total']], fill_value = 0)
    
final = totals.merge(signals, left_index = True, right_index = True)

# Visualize
plt.figure(figsize = (50, 20))
plt.plot(final['total'])
plt.title('30-Day/90-Day Short Trading for S&P 500: 2000 - 2022', fontsize = 24)
plt.xlabel('Date', fontsize = 14)
plt.ylabel('Total Portfolio $ (Million)', fontsize = 14)
        
'''
    Efficient Frontier
'''

# Build df
stocks = pd.DataFrame()

for nm, grp in df1.groupby('ticker'):
    
    stocks[nm] = grp['price']

# Generate portfolii
returns = stocks.pct_change()
meanreturns = returns.mean()
cov_mat = returns.cov()

# Get optima
optima = display_simulated_allocation_report(meanreturns, cov_mat, 100000, .02)
assets = optima.columns
allocations = []

for asset in optima.columns:
    
    if optima[asset]['allocation'] > 0:
        
        print(f"Asset: {asset}, allocation = {optima[asset]['allocation']}")
        
        allocations.append((asset, optima[asset]['allocation']))

# Inidivudally iterate to determine allocations
starting_cash = 10000 * 503
total_shares = 1000 * 503
frontier = pd.DataFrame()

# Iterate efficient frontier to calculate new totals       
for item in tqdm.tqdm(allocations):
    
    data = df1[df1['ticker'] == item[0]]
    
    positions = pd.DataFrame(index = data.index).fillna(0.0)
    positions[nm] = 1000 * 503 * item[1] * data['signal'] 

    port = positions.multiply(data['price'], axis = 0)
    pos_diff = positions.diff()

    # Compute assets
    port['holdings'] = (positions.multiply(data['price'], axis = 0)).sum(axis = 1)
    port['cash'] = 10000 - (pos_diff.multiply(data['price'], axis = 0)).sum(axis = 1).cumsum()   
    port['total'] = port['cash'] + port['holdings']
    port['returns'] = port['total'].pct_change() 
    port['ticker'] = item[0]
    
    frontier = frontier.append(port)
    
# Aggregate portfolio returns
efficientfrontier = frontier.groupby(frontier.index)['holdings', 'cash', 'total'].agg({'holdings': 'sum',
                                                                                       'cash': 'sum',
                                                                                       'total': 'sum'}, axis = 1)

# Visualize
plt.figure(figsize = (50, 20))
plt.plot(efficientfrontier['total'], label = 'Efficient Frontier')
plt.plot(final['total'], label = 'Baseline Trading')
plt.title('Efficient Frontier 30-Day/90-Day Short Trading for S&P 500: 2000 - 2022', fontsize = 24)
plt.xlabel('Date', fontsize = 14)
plt.ylabel('Total Portfolio $ (Million)', fontsize = 14)

# Evaluate
print(f"Efficient Frontier Portfolio starting cash:  ${efficientfrontier['total'].iloc[0]}")
print(f"Efficient Frontier Portfolio ending cash:    ${round(efficientfrontier['total'].iloc[-1])}")
print(f"Efficient Frontier overall returns: {round((efficientfrontier['total'].iloc[-1]/efficientfrontier['total'].iloc[0]))}X")

# Tabulate categories
tickers.groupby(['GICS Sector', 'GICS Sub-Industry']).size().reset_index(name = 'count').sort_values('count', ascending = False)
tickers['category'] = tickers['GICS Sector'] + ' / ' + tickers['GICS Sub-Industry']

# Coalesce results by sector
sector_totals = pd.DataFrame()

for nm, grp in tickers.groupby('category'): 
    
    print(nm)
    
    total = pd.DataFrame()
    
    for ticker in grp['Symbol'].unique():
        
        total = total.add(frontier[frontier['ticker'] == ticker][['holdings', 'cash', 'total']], fill_value = 0)
        
    total['category'] = nm
    
    sector_totals = sector_totals.append(total)

sector_stats = pd.DataFrame()
    
for nm, grp in sector_totals.groupby('category'):
    
    sector_stats = sector_stats.append({'category': nm,
                                        'start_dt': min(grp.index.date),
                                        'end_dt': max(grp.index.date),
                                        'time_in_years': round((max(grp.index.date) - min(grp.index.date)).days/365, 3),
                                        'total_size': tickers[tickers['category'] == nm].size,
                                        'start_investment': tickers[tickers['category'] == nm].size * 10000,
                                        'end_investment': round(grp['total'].iloc[-1], 2),
                                        'overall_return': round(grp['total'].iloc[-1], 2) / (tickers[tickers['category'] == nm].size * 10000)}, ignore_index = True)
    
    print(f'Analysis for {nm}: ')
    print(f'Investment Start Date: {min(grp.index.date)}')
    print(f'Total Length of Investment: {round((max(grp.index.date) - min(grp.index.date)).days/365, 3)} years')
    print(f"Num. Companies: {tickers[tickers['category'] == nm].size}")
    print(f"Starting Investment: ${tickers[tickers['category'] == nm].size * 10000}")
    print(f"Ending Investment:   ${round(grp['total'].iloc[-1], 2)}")
    print(f"Overall Returns for Sector: {round(grp['total'].iloc[-1] / (tickers[tickers['category'] == nm].size * 10000), 3)}X")
    print()

bad_co = 0
good_co = 0

for idx, row in sector_stats.sort_values('overall_return').iterrows():
    
    if row['overall_return'] <= 1:
        
        pass
    
        # print(f"Bad Investment Category: {row['category']}")
        # print(f"Related Companies:")
        # print(', '.join([i for i in tickers[tickers['category'] == row['category']]['Symbol']]))
        
        # for i in tickers[tickers['category'] == row['category']]['Symbol']:
            
        #     bad_co += 1
            
        # print()
        
    else:
        
        print(f"Analysis for {row['category']}: ")
        print(f"Investment Start Date: {row['start_dt']}")
        print(f"Total Length of Investment: {row['time_in_years']} years")
        print(f"Size of Sector Portfolio- {tickers[tickers['category'] == row['category']]['Symbol'].count()} companies")
        print(f"Related Companies: {', '.join([i for i in tickers[tickers['category'] == row['category']]['Symbol']])}")
        print(f"Overall Return on Investment: {round(sector_totals[sector_totals['category'] == row['category']]['total'].iloc[-1] / sector_totals[sector_totals['category'] == row['category']]['total'].iloc[0], 3)}X")
        
        for i in tickers[tickers['category'] == row['category']]['Symbol']:

            good_co += 1
            
        print()  
        