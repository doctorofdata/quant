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

# Classical MPT
class markowitz_portfolio:
    
    def __init__(self, df):
        
        # Initialize    
        self.tbl = df[['price', 'ticker']].pivot(columns = 'ticker')
        self.tbl.columns = [col[1] for col in self.tbl.columns]
        self.ret = self.tbl.pct_change()
        self.mean_ret = self.ret.mean()
        self.cov_matrix = self.ret.cov()

        # Constants
        self.num_portfolios = 50000
        self.risk_free_rate = 0.0178       
        
        # Call
        self.opt_weights = self.display_simulations(self.mean_ret, self.cov_matrix, self.num_portfolios, self.risk_free_rate) 

    # Calculate
    def annual_performance(self, weights, mean_returns, cov_matrix):
    
        returns = np.sum(mean_returns * weights ) * 252
        std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    
        return std, returns
  
    def simulator(self, num_portfolios, mean_ret, cov_matrix, risk_free_rate):
    
        results = np.zeros((38, num_portfolios))
        weights_record = []
    
        for i in range(num_portfolios):
        
            weights = np.random.random(38)
            weights /= np.sum(weights)
            weights_record.append(weights)
            portfolio_std_dev, portfolio_return = self.annual_performance(weights, mean_ret, cov_matrix)
            results[0,i] = portfolio_std_dev
            results[1,i] = portfolio_return
            results[2,i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
        
        return results, weights_record

    def display_simulations(self, mean_ret, cov_matrix, num_portfolios, risk_free_rate):
    
        results, weights = self.simulator(num_portfolios, mean_ret, cov_matrix, risk_free_rate)
    
        max_sharpe_idx = np.argmax(results[2])
        
        sdp, rp = results[0,max_sharpe_idx], results[1,max_sharpe_idx]
        
        max_sharpe_allocation = pd.DataFrame(weights[max_sharpe_idx],index = self.tbl.columns, columns = ['allocation'])
        max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
        max_sharpe_allocation = max_sharpe_allocation.T
    
        min_vol_idx = np.argmin(results[0])
        
        sdp_min, rp_min = results[0,min_vol_idx], results[1,min_vol_idx]
        
        min_vol_allocation = pd.DataFrame(weights[min_vol_idx],index = self.tbl.columns, columns = ['allocation'])
        min_vol_allocation.allocation = [round(i*100,2)for i in min_vol_allocation.allocation]
        min_vol_allocation = min_vol_allocation.T
        
        print("-"*80)
        print("Maximum Sharpe Ratio Portfolio Allocation\n")
        print("Annualised Return:", round(rp,2))
        print("Annualised Volatility:", round(sdp,2))
        print("\n")
        print(max_sharpe_allocation)
        print("-"*80)
        print("Minimum Volatility Portfolio Allocation\n")
        print("Annualised Return:", round(rp_min,2))
        print("Annualised Volatility:", round(sdp_min,2))
        print("\n")
        print(min_vol_allocation)
        
        plt.figure(figsize = (10, 7))
        plt.scatter(results[0,:], results[1,:], c = results[2,:], cmap = 'YlGnBu', marker = 'o', s = 10, alpha = 0.3)
        plt.colorbar()
        plt.scatter(sdp,rp,marker='*',color='r',s=500, label='Maximum Sharpe ratio')
        plt.scatter(sdp_min,rp_min,marker='*',color='g',s=500, label='Minimum volatility')
        plt.title('Simulated Portfolio Optimization based on Efficient Frontier', fontsize = 12)
        plt.xlabel('annualised volatility', fontsize = 12)
        plt.ylabel('annualised returns', fontsize = 12)
        plt.legend(labelspacing=0.8)
    
        return max_sharpe_allocation

    def efficient_return(self, mean_returns, cov_matrix, target):
    
        num_assets = len(mean_returns)
        args = (mean_returns, cov_matrix)

        def portfolio_return(weights):
        
            return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[1]

        constraints = ({'type': 'eq', 'fun': lambda x: portfolio_return(x) - target},
                       {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
        bounds = tuple((0,1) for asset in range(num_assets))
        result = sco.minimize(portfolio_volatility, num_assets * [1./num_assets,], args = args, method = 'SLSQP', bounds = bounds, constraints = constraints)
    
        return result

    def efficient_frontier(self, mean_returns, cov_matrix, returns_range):
    
        efficients = []
    
        for ret in returns_range:
        
            efficients.append(efficient_return(mean_returns, cov_matrix, ret))
        
        return efficients
    
    
    
    
    
    
    
    
    
    