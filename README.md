# Scalable Distributed Deep Reinforcement Learning for Multiple Stock Trading

This project implements a scalable distributed deep reinforcement learning solution for automated trading of multiple stocks. It's based on the FinRL framework and uses the IMPALA (Importance Weighted Actor-Learner Architectures) algorithm for scalable distributed reinforcement learning.

## Introduction

Deep reinforcement learning (Deep RL) combines reinforcement learning and deep learning, offering powerful solutions for complex decision-making tasks. This project applies Deep RL to the challenging domain of automated stock trading, aiming to create a scalable and efficient solution for trading multiple stocks simultaneously.

## Features

- Automated trading of DOW-30 stocks
- Scalable distributed reinforcement learning using IMPALA
- Integration of technical indicators (MACD, RSI) and turbulence index
- Backtesting and performance evaluation

## Requirements

- Python 3
- Packages:
  - FinRL
  - Yahoo Finance API
  - pandas
  - numpy
  - matplotlib
  - stockstats
  - OpenAI gym
  - stable-baselines
  - tensorflow
  - pyfolio
  - jupyter

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the Jupyter Notebook `main.ipynb` to execute the project.

## Implementation Details

### Data Preprocessing

- Data source: Yahoo Finance API
- Time range: 2009-01-01 to 2021-10-31
- Features: Open, High, Low, Close, Volume, Dividends, Stock Splits
- Additional indicators: MACD, RSI, Turbulence Index

### Environment Design

- Based on OpenAI Gym framework
- Action space: -k to +k (k = number of shares, negative for selling, positive for buying, 0 for holding)
- Reward function: Difference in portfolio values between states
- State space: Current market observations

### Distributed Deep RL Algorithm

- IMPALA (Importance Weighted Actor-Learner Architectures)
- V-trace off-policy actor-critic algorithm

## Results

When tested with an initial capital of $1,000,000 on 2020-07-01:

- Average annual return: 30.35%
- Total cumulative return: 42.40%
- Outperformed baseline by approximately 5%

For detailed performance metrics and visualizations, please refer to the full project report.

## Contributors

- Tianzhang Cai
- Yage Hao

## Acknowledgements

This project was completed as part of the ID2223: Scalable Machine Learning and Deep Learning course.

## References

1. FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance (NeurIPS 2020: Deep RL Workshop)
2. IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures
3. Yahoo Finance API
