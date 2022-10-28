from statsmodels.graphics.tsaplots import acf,pacf,plot_acf,plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import pmdarima as pm
import matplotlib.pyplot as plt
import pandas as pd

class ARIMA:
    def __init__(self,x):
        self.x = x

    """
    对数据进行测试，返回报告
    """
    def summary(self,x):
        arima = pm.auto_arima(x, error_action='ignore', trace=True, seasonal=False)
        return arima

    """
    稳定性判断
    """
    def test_stationarity(self,timeseries, window=7):
        rolmean = timeseries.rolling(window).mean()
        rolstd = timeseries.rolling(window).std()

        # plot rolling statistics:
        plt.figure(figsize=(15, 5))
        plt.plot(timeseries, label='Original')
        plt.plot(rolmean, label='Rolling Mean')
        plt.plot(rolstd, label='Rolling Standard Deviation')
        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation')
        plt.grid()

        # Dickey-Fuller test:
        print('Results of Augmented Dickey-Fuller Test:')
        dftest = adfuller(timeseries, autolag='AIC')
        # dftest的输出前一项依次为检测值，p值，滞后数，使用的观测数，各个置信度下的临界值
        dfoutput = pd.Series(dftest[0:4],
                             index=['Test Statistic', 'P-Value', 'Lags Used', 'Number of Observations Used'])
        for key, value in dftest[4].items():
            dfoutput['Critical value (%s)' % key] = value
        return dfoutput

    """
    绘制自相关图
    """
    def draw_acf_pacf(self,ts, lags):
        f = plt.figure(figsize=(15, 5))
        ax1 = f.add_subplot(211)
        plot_acf(ts, ax=ax1, lags=lags)
        plt.grid()
        ax2 = f.add_subplot(212)
        plot_pacf(ts, ax=ax2, lags=lags)
        plt.subplots_adjust(hspace=0.5)
        plt.grid()

    """
    建模
    """
    def arima(self,x,p,d,q):
        modle = ARIMA(x, order=(p, d, q))
        result = modle.fit()
        return result



