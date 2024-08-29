# SPY-Trend-Prediction-with-Deep-Learning-Models
Overview

This project is part of the Certificate in Quantitative Finance (CQF) Final Project, focused on developing a robust predictive model using Long Short-Term Memory (LSTM) networks to forecast daily uptrends in the SPY ETF, which tracks the S&P 500 index. The goal is to create a highly effective trading strategy that outperforms the market while managing risks efficiently.

Project Structure

Code: Python scripts for data processing, feature engineering, model building, and backtesting.
Data: Historical financial data including SPY ETF prices, technical indicators, Treasury yields, and other asset data from July 2017 to July 2024.
Documentation: Detailed project report and supplementary documentation.
Features

1. Feature Engineering:
Data Sources: Collected daily OHLC data for the SPY ETF, technical indicators, other assets data (GOLD, UUP, USO, VIX) and macroeconomic data (e.g., 2-Year and 10-Year Treasury Yields).
Time Period: July 2017 to July 2024.
Feature Reduction: Reduced 355 initial features to 24 key features using Recursive Feature Elimination (RFE) and Self-Organizing Maps (SOM) to enhance model performance.

2. Exploratory Data Analysis (EDA):
Conducted comprehensive EDA to identify relationships between features and the target variable.
Addressed class imbalance by defining the target based on a 0.20% threshold in daily returns.

3. Data Handling & Sequence Generation:
Data Handling: Processed the dataset by ensuring all features were properly scaled using Min-Max scaling. This step was crucial for preparing the data for input into the LSTM models.
Sequence Generation: Generated sequences with a 21-day lookback period, where each sequence consists of 21 consecutive days of historical data. This data was reshaped into a 3D format (samples, time steps, features) to align with the input requirements of LSTM models. This step is essential for capturing temporal dependencies in the data, which are critical for making accurate predictions.

4. Model Building & Optimization:
Models Built:
Single-Layer LSTM: Achieved the highest accuracy (89.32%) and recall (0.9921), making it the best-performing model.
Two-Layer LSTM: Improved sequential modeling with two LSTM layers.
Five-Layer LSTM: A deeper model architecture for capturing intricate patterns.
GRU (Gated Recurrent Unit): An alternative architecture providing competitive performance.
LSTM + CNN (Convolutional Neural Network): A hybrid model combining Conv1D layers with LSTM to capture both spatial and temporal patterns.

Hyperparameter Tuning: Used Random Search to optimize model parameters, including units, dropout rates, and learning rates.

5. Model Evaluation:
Evaluated each model using metrics such as accuracy, precision, recall, F1 score, and AUC.
Selected the Single-Layer LSTM as the final model due to its superior balance of accuracy and recall.

6. Trading Strategy & Backtesting:
Strategy Development: Based on LSTM model predictions, incorporating risk management techniques like stop-loss orders and take-profit targets.
Performance Metrics:
Cumulative Return: 39.13%, outperforming the SPY ETF benchmark return of 34.34%.
Compound Annual Growth Rate (CAGR): 20.67% vs. the benchmark’s 18.28%.
Sharpe Ratio: 4.09, indicating superior risk-adjusted returns compared to the benchmark’s 2.21.
Maximum Drawdown: 2.08%, significantly lower than the benchmark’s 10.54%.
Volatility (Annualized): 6.74%, compared to the benchmark’s 11.34%.
Sortino Ratio: Achieved a ratio of 10.66 versus 3.5 for the benchmark, indicating effective downside risk management.

7. Tools & Technologies:
Python Libraries: TensorFlow, Keras, Scikit-Learn, Pandas, QuantStats.
Visualization: Utilized TensorBoard for visualizing hyperparameter tuning and model training progress.
