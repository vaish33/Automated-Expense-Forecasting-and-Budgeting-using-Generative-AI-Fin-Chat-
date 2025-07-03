import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    data = data.asfreq('MS').ffill()
    return data

def check_stationarity(data, column):
    data_series = data[column].dropna()
    if len(data_series) < 2:
        raise ValueError(f"Not enough data to check stationarity for column: {column}")
    result = adfuller(data_series)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    return result[1] > 0.05

def make_stationary(data, column):
    if check_stationarity(data, column):
        data[column + '_diff'] = data[column].diff().dropna()
        return data[column + '_diff'].dropna()
    return data[column]

if __name__ == "__main__":
    data = load_and_preprocess_data('expense_data_it_project.csv')
    stationary_data = make_stationary(data, 'development_cost')
    print(stationary_data.head())
