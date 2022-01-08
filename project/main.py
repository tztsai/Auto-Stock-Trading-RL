# %% [markdown]
# <a href="https://colab.research.google.com/github/AI4Finance-Foundation/FinRL/blob/master/FinRL_StockTrading_NeurIPS_2018.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # Deep Reinforcement Learning for Stock Trading from Scratch: Multiple Stock Trading
# 
# * **Pytorch Version** 
# 
# 

# %% [markdown]
# # Content

# %% [markdown]
# * [1. Problem Definition](#0)
# * [2. Getting Started - Load Python packages](#1)
#     * [2.1. Install Packages](#1.1)    
#     * [2.2. Check Additional Packages](#1.2)
#     * [2.3. Import Packages](#1.3)
#     * [2.4. Create Folders](#1.4)
# * [3. Download Data](#2)
# * [4. Preprocess Data](#3)        
#     * [4.1. Technical Indicators](#3.1)
#     * [4.2. Perform Feature Engineering](#3.2)
# * [5.Build Environment](#4)  
#     * [5.1. Training & Trade Data Split](#4.1)
#     * [5.2. User-defined Environment](#4.2)   
#     * [5.3. Initialize Environment](#4.3)    
# * [6.Implement DRL Algorithms](#5)  
# * [7.Backtesting Performance](#6)  
#     * [7.1. BackTestStats](#6.1)
#     * [7.2. BackTestPlot](#6.2)   
#     * [7.3. Baseline Stats](#6.3)   
#     * [7.3. Compare to Stock Market Index](#6.4)   
# * [RLlib Section](#7)            

# %% [markdown]
# <a id='0'></a>
# # Part 1. Problem Definition

# %% [markdown]
# This problem is to design an automated trading solution for single stock trading. We model the stock trading process as a Markov Decision Process (MDP). We then formulate our trading goal as a maximization problem.
# 
# The algorithm is trained using Deep Reinforcement Learning (DRL) algorithms and the components of the reinforcement learning environment are:
# 
# 
# * Action: The action space describes the allowed actions that the agent interacts with the
# environment. Normally, a ∈ A includes three actions: a ∈ {−1, 0, 1}, where −1, 0, 1 represent
# selling, holding, and buying one stock. Also, an action can be carried upon multiple shares. We use
# an action space {−k, ..., −1, 0, 1, ..., k}, where k denotes the number of shares. For example, "Buy
# 10 shares of AAPL" or "Sell 10 shares of AAPL" are 10 or −10, respectively
# 
# * Reward function: r(s, a, s′) is the incentive mechanism for an agent to learn a better action. The change of the portfolio value when action a is taken at state s and arriving at new state s',  i.e., r(s, a, s′) = v′ − v, where v′ and v represent the portfolio
# values at state s′ and s, respectively
# 
# * State: The state space describes the observations that the agent receives from the environment. Just as a human trader needs to analyze various information before executing a trade, so
# our trading agent observes many different features to better learn in an interactive environment.
# 
# * Environment: Dow 30 consituents
# 
# 
# The data of the single stock that we will be using for this case study is obtained from Yahoo Finance API. The data contains Open-High-Low-Close price and volume.
# 

# %% [markdown]
# <a id='1'></a>
# # Part 2. Getting Started- Load Python Packages

# %% [markdown]
# <a id='1.1'></a>
# ## 2.1. Install all the packages through FinRL library
# 

# %%


# %% [markdown]
# 
# <a id='1.2'></a>
# ## 2.2. Check if the additional packages needed are present, if not install them. 
# * Yahoo Finance API
# * pandas
# * numpy
# * matplotlib
# * stockstats
# * OpenAI gym
# * stable-baselines
# * tensorflow
# * pyfolio

# %% [markdown]
# <a id='1.3'></a>
# ## 2.3. Import Packages

# %%
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import datetime

from apps import config
from meta.preprocessor.yahoodownloader import YahooDownloader
from meta.preprocessor.preprocessors import FeatureEngineer, data_split
from meta.env_stock_trading.env_stocktrading import StockTradingEnv
# from drl_agents.stablebaselines3.models import DRLAgent
from drl_agents.torchbeast import monobeast
from meta.data_processor import DataProcessor

from plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
from pprint import pprint
from absl import flags

import sys
sys.path.append("../FinRL-Library")

import itertools

FLAGS = flags.FLAGS
FLAGS(sys.argv, known_only=True)

# %% [markdown]
# <a id='1.4'></a>
# ## 2.4. Create Folders

# %%
import os
if not os.path.exists("./" + config.DATA_SAVE_DIR):
    os.makedirs("./" + config.DATA_SAVE_DIR)
if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
    os.makedirs("./" + config.TRAINED_MODEL_DIR)
if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
    os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
if not os.path.exists("./" + config.RESULTS_DIR):
    os.makedirs("./" + config.RESULTS_DIR)

# %% [markdown]
# <a id='2'></a>
# # Part 3. Download Data
# Yahoo Finance is a website that provides stock data, financial news, financial reports, etc. All the data provided by Yahoo Finance is free.
# * FinRL uses a class **YahooDownloader** to fetch data from Yahoo Finance API
# * Call Limit: Using the Public API (without authentication), you are limited to 2,000 requests per hour per IP (or up to a total of 48,000 requests a day).
# 

# %% [markdown]
# 
# 
# -----
# class YahooDownloader:
#     Provides methods for retrieving daily stock data from
#     Yahoo Finance API
# 
#     Attributes
#     ----------
#         start_date : str
#             start date of the data (modified from config.py)
#         end_date : str
#             end date of the data (modified from config.py)
#         ticker_list : list
#             a list of stock tickers (modified from config.py)
# 
#     Methods
#     -------
#     fetch_data()
#         Fetches data from yahoo API
# 

# %%
# from config.py start_date is a string
config.START_DATE

# %%
# from config.py end_date is a string
config.END_DATE

# %%
print(config.DOW_30_TICKER)

# %%
data_filename = 'processed_data.csv'
data_path = os.path.join(config.DATA_SAVE_DIR, data_filename)

if os.path.exists(data_path):
    data = pd.read_csv(data_path)
else:
    data = None

# %%
if data is None:
    df = YahooDownloader(start_date = '2009-01-01',
                     end_date = '2021-10-31',
                     ticker_list = config.DOW_30_TICKER).fetch_data()

# %%
if data is None: df.shape

# %%
if data is None: df.sort_values(['date','tic'],ignore_index=True).head()

# %% [markdown]
# # Part 4: Preprocess Data
# Data preprocessing is a crucial step for training a high quality machine learning model. We need to check for missing data and do feature engineering in order to convert the data into a model-ready state.
# * Add technical indicators. In practical trading, various information needs to be taken into account, for example the historical stock prices, current holding shares, technical indicators, etc. In this article, we demonstrate two trend-following technical indicators: MACD and RSI.
# * Add turbulence index. Risk-aversion reflects whether an investor will choose to preserve the capital. It also influences one's trading strategy when facing different market volatility level. To control the risk in a worst-case scenario, such as financial crisis of 2007–2008, FinRL employs the financial turbulence index that measures extreme asset price fluctuation.

# %%
if data is None:
    fe = FeatureEngineer(
                        use_technical_indicator=True,
                        tech_indicator_list = config.TECHNICAL_INDICATORS_LIST,
                        use_vix=True,
                        use_turbulence=True,
                        user_defined_feature = False)

    processed = fe.preprocess_data(df)

# %%
if data is None:
    list_ticker = processed["tic"].unique().tolist()
    list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str))
    combination = list(itertools.product(list_date,list_ticker))

    processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left")
    processed_full = processed_full[processed_full['date'].isin(processed['date'])]
    processed_full = processed_full.sort_values(['date','tic'])

    processed_full = processed_full.fillna(0)
    processed_full.to_csv(data_path, index=False)
    
else:
    processed_full = data

# %%
processed_full.sort_values(['date','tic'],ignore_index=True).head(10)

# %% [markdown]
# <a id='4'></a>
# # Part 5. Design Environment
# Considering the stochastic and interactive nature of the automated stock trading tasks, a financial task is modeled as a **Markov Decision Process (MDP)** problem. The training process involves observing stock price change, taking an action and reward's calculation to have the agent adjusting its strategy accordingly. By interacting with the environment, the trading agent will derive a trading strategy with the maximized rewards as time proceeds.
# 
# Our trading environments, based on OpenAI Gym framework, simulate live stock markets with real market data according to the principle of time-driven simulation.
# 
# The action space describes the allowed actions that the agent interacts with the environment. Normally, action a includes three actions: {-1, 0, 1}, where -1, 0, 1 represent selling, holding, and buying one share. Also, an action can be carried upon multiple shares. We use an action space {-k,…,-1, 0, 1, …, k}, where k denotes the number of shares to buy and -k denotes the number of shares to sell. For example, "Buy 10 shares of AAPL" or "Sell 10 shares of AAPL" are 10 or -10, respectively. The continuous action space needs to be normalized to [-1, 1], since the policy is defined on a Gaussian distribution, which needs to be normalized and symmetric.

# %% [markdown]
# ## Training data split: 2009-01-01 to 2020-07-01
# ## Trade data split: 2020-07-01 to 2021-10-31

# %%
train = data_split(processed_full, '2009-01-01','2020-07-01')
trade = data_split(processed_full, '2020-07-01','2021-10-31')
print(len(train))
print(len(trade))

# %%
train.tail()

# %%
trade.head()

# %%
config.TECHNICAL_INDICATORS_LIST

# %%
stock_dimension = len(train.tic.unique())
state_space = 1 + 2*stock_dimension + len(config.TECHNICAL_INDICATORS_LIST)*stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")


# %%
env_kwargs = {
    "hmax": 100, 
    "initial_amount": 1000000, 
    "buy_cost_pct": 0.001,
    "sell_cost_pct": 0.001,
    "state_space": state_space, 
    "stock_dim": stock_dimension, 
    "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST, 
    "action_space": stock_dimension, 
    "reward_scaling": 1e-4
    
}

e_train_gym = StockTradingEnv(df = train, **env_kwargs)

# %% [markdown]
# ## Environment for Training
# 
# 

# %%
env_train, _ = e_train_gym.get_sb_env()
print(type(env_train))

# %%
monobeast.CustomEnv.env = e_train_gym
monobeast.train(FLAGS)

# %% [markdown]
# ## Trading
# Assume that we have $1,000,000 initial capital at 2020-07-01. We use the DDPG model to trade Dow jones 30 stocks.

# %% [markdown]
# ### Set turbulence threshold
# Set the turbulence threshold to be greater than the maximum of insample turbulence data, if current turbulence index is greater than the threshold, then we assume that the current market is volatile

# %%
data_risk_indicator = processed_full[(processed_full.date<'2020-07-01') & (processed_full.date>='2009-01-01')]
insample_risk_indicator = data_risk_indicator.drop_duplicates(subset=['date'])

# %%
insample_risk_indicator.vix.describe()

# %%
insample_risk_indicator.vix.quantile(0.996)

# %%
insample_risk_indicator.turbulence.describe()

# %%
insample_risk_indicator.turbulence.quantile(0.996)

# %% [markdown]
# ### Trade
# 
# DRL model needs to update periodically in order to take full advantage of the data, ideally we need to retrain our model yearly, quarterly, or monthly. We also need to tune the parameters along the way, in this notebook I only use the in-sample data from 2009-01 to 2020-07 to tune the parameters once, so there is some alpha decay here as the length of trade date extends. 
# 
# Numerous hyperparameters – e.g. the learning rate, the total number of samples to train on – influence the learning process and are usually determined by testing some variations.

# %%
#trade = data_split(processed_full, '2020-07-01','2021-10-31')
e_trade_gym = StockTradingEnv(df = trade, turbulence_threshold = 70,risk_indicator_col='vix', **env_kwargs)
# env_trade, obs_trade = e_trade_gym.get_sb_env()

# %%
trade.head()

# %%
df_account_value, df_actions = DRLAgent.DRL_prediction(
    model=trained_sac, 
    environment = e_trade_gym)

# %%
df_account_value.shape

# %%
df_account_value.tail()

# %%
df_actions.head()

# %% [markdown]
# <a id='6'></a>
# # Part 7: Backtest Our Strategy
# Backtesting plays a key role in evaluating the performance of a trading strategy. Automated backtesting tool is preferred because it reduces the human error. We usually use the Quantopian pyfolio package to backtest our trading strategies. It is easy to use and consists of various individual plots that provide a comprehensive image of the performance of a trading strategy.

# %% [markdown]
# <a id='6.1'></a>
# ## 7.1 BackTestStats
# pass in df_account_value, this information is stored in env class
# 

# %%
print("==============Get Backtest Results===========")
now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')

perf_stats_all = backtest_stats(account_value=df_account_value)
perf_stats_all = pd.DataFrame(perf_stats_all)
perf_stats_all.to_csv("./"+config.RESULTS_DIR+"/perf_stats_all_"+now+'.csv')

# %%
#baseline stats
print("==============Get Baseline Stats===========")
baseline_df = get_baseline(
        ticker="^DJI", 
        start = df_account_value.loc[0,'date'],
        end = df_account_value.loc[len(df_account_value)-1,'date'])

stats = backtest_stats(baseline_df, value_col_name = 'close')


# %%
df_account_value.loc[0,'date']

# %%
df_account_value.loc[len(df_account_value)-1,'date']

# %% [markdown]
# <a id='6.2'></a>
# ## 7.2 BackTestPlot

# %%
print("==============Compare to DJIA===========")

# S&P 500: ^GSPC
# Dow Jones Index: ^DJI
# NASDAQ 100: ^NDX
backtest_plot(df_account_value, 
             baseline_ticker = '^DJI', 
             baseline_start = df_account_value.loc[0,'date'],
             baseline_end = df_account_value.loc[len(df_account_value)-1,'date'])
