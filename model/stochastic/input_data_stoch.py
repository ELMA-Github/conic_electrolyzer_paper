import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from matplotlib import rcParams


## Import time series dataset

# Read CSV file
data = pd.read_csv('data/11092019.csv')

# Convert date into datatime
data["Date"] = pd.to_datetime(data["Datetime"], format="%d/%m/%Y %H:%M", errors="coerce")
mask = data.Date.isnull()
data.loc[mask, 'Date']= pd.to_datetime(data[mask]['Datetime'], format='%d-%m-%Y %H:%M',errors='coerce')

# Add comumn with year, month, day, hour
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
data['Hour'] = data['Date'].dt.hour

# Set datetime column as index
data.set_index('Date', inplace = True)


def data_CP_w(N_t):
    arr = data[['CP']].to_numpy().flatten()
    # change the dtype to 'float64'
    arr = arr.astype('float64')
    arr = arr[:N_t]
    return arr
    
def data_pi_e(N_t):
    arr = data[['El_price_EUR_MWh']].to_numpy().flatten()
    # change the dtype to 'float64'
    arr = arr.astype('float64')
    arr = arr[:N_t]
    return arr