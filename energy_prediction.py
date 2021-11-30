import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

import tensorflow as tf
import numpy as np
import os
import shutil
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import ml_metrics as metrics

from fbprophet import Prophet


#%% Load data

data_sheets = [
                # 'A1_no_negatives.xlsx',
                # 'A1.xlsx',
                # 'A2.xlsx',\
                # 'A3.xlsx',\
                'A4.xlsx'
               ]

options = [
            # 'Main Data',\
            # 'A2',\
           # 'A3',\
           'A4'
          ]

for sheet in data_sheets:
  for opt in options:

    data_to_input = sheet
    option = opt

    data_name = data_to_input[0:-5] + '_' + str(option)
    if os.path.exists('.' + os.sep + data_name):
        shutil.rmtree('.' + os.sep + data_name)
    os.mkdir('.' + os.sep + data_name)
    current_dir = '.' + os.sep + data_name

    print('Loading data...please wait')
    df = pd.read_excel(data_to_input, str(option), convert_float=True,
                       dtype={'a': np.float64, 'b': np.int32})

    print("DATA BEING ANALYZED: ", data_to_input)
    print("TAB: ", opt)

    print('------------------------------------------------------------------')

df['date'] +=  pd.to_timedelta(df.hour, unit='h')
df = df.sort_values(by=['date'])

data = df
data = data.filter(items = ['date', 'Energy (kWh)'])

# scaler = StandardScaler()
# data = scaler.fit_transform(data)

data = data.rename(columns={ "Energy (kWh)": "energy"})
data = data.set_index('date')

y = data['energy']
y = y.resample('MS').mean()


#%% Data Decomposition
from pylab import rcParams
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(y, model='additive', period = 30)
fig = decomposition.plot()
plt.show()
plt.savefig(current_dir + os.sep + 'DATA_DECOMPOSITION.png')
plt.close()

#%% Fit an ARIMA
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

warnings.filterwarnings("ignore")
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(endog=train.Rides,\
                                            trend='n',\
                                            order=(1,0,1),\
                                            seasonal_order=(1,0,1,12))

            results = mod.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue

mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 0, 1),
                                seasonal_order=(1, 0, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])

#%% Model Diagnostics

results.plot_diagnostics(figsize=(16, 8))
plt.show()
plt.savefig(current_dir + os.sep + 'SARIMAX_DIAGNOSTICS.png')
plt.close()

#%% Validating Forecast
start_forecast = 100
pred = results.get_prediction(start=start_forecast, dynamic=False)
pred_ci = pred.conf_int()

ax = y.plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='Predictions', alpha=.7)

ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Date')
ax.set_ylabel('Energy')
plt.legend()

plt.show()
plt.savefig(current_dir + os.sep + 'SARIMAX_VALIDATION.png')
plt.close()

y_forecasted = pred.predicted_mean
y_truth = y[100:]

#%% Accuracy metrics

# Accuracy metrics
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    mse = mean_squared_error(actual,forecast) # mse
    r2 = r2_score(actual,forecast) #r2
    print({'mape':mape, 'rmse':rmse, 'mse':mse, 'r2':r2})
    return({mape, rmse, mse, r2})

mape, rmse, mse, r2 = forecast_accuracy(y_forecasted, y_truth)

metrics = pd.DataFrame(data = (mape, rmse, mse, r2),\
                        index = ('mape', 'rmse', 'mse', 'r2'))
metrics.to_csv(current_dir + os.sep + 'SARIMAX_METRICS.csv')

#%% Forecast 1y
years = 1
pred_uc = results.get_forecast(steps=12*years)
pred_ci = pred_uc.conf_int()
ax = y.plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Energy')
plt.legend()
plt.show()
plt.savefig(current_dir + os.sep + 'SARIMAX_FORECAST_1y.png')
plt.close()

#%% Forecast 5y
years = 5
pred_uc = results.get_forecast(steps=12*years)
pred_ci = pred_uc.conf_int()
ax = y.plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Energy')
plt.legend()
plt.show()
plt.savefig(current_dir + os.sep + 'SARIMAX_FORECAST_5y.png')
plt.close()

#%% Data Pre-processing
data = df
data = data.filter(items = ['date', 'Energy (kWh)'])

data = data.rename(columns={ "Energy (kWh)": "energy"})
data = data.set_index('date')

y = data['energy']
y = y.resample('W').mean()

data = y.to_frame()
data = data.reset_index()

data = data.rename(columns={"date": "ds", "energy": "y"})

#%% Fit a PROPHET
print('Training a prophet...')
m = Prophet()
m.fit(data)

#%% Forecast 1 year
years = 1
future = m.make_future_dataframe(periods=365*years)
future.tail()

forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

fig1 = m.plot(forecast)
plt.savefig(current_dir + os.sep + 'PROPHET_FORECAST_1y.png')
plt.close()

fig2 = m.plot_components(forecast)
plt.savefig(current_dir + os.sep + 'PROPHET_COMPONENTS.png')
plt.close()

#%% Forecast 5 years
years = 5
future = m.make_future_dataframe(periods=365*years)
future.tail()

forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

fig1 = m.plot(forecast)
plt.savefig(current_dir + os.sep + 'PROPHET_FORECAST_5y.png')
plt.close()

#%% Accuracy Metrics

metric_df = forecast.set_index('ds')[['yhat']].join(data.set_index('ds').y)\
  .dropna().reset_index()

mape, rmse, mse, r2 = forecast_accuracy(metric_df.yhat,  metric_df.y)

metrics = pd.DataFrame(data = (mape, rmse, mse, r2),\
                        index = ('mape', 'rmse', 'mse', 'r2'))
metrics.to_csv(current_dir + os.sep + 'PROPHET_METRICS.csv')
