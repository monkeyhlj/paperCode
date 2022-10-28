__all__ = ["ICEEMDAN"]

import matlab.engine
eng = matlab.engine.start_matlab()
import matplotlib.pyplot as plt
import numpy as np

class ICEEMDAN:
    def __init__(self,df):
        self.df = df

    """
    ICEEMDAN进行分解
    :param df:需要分解的数据
    """
    def iceemdan(self,df):
        df = np.array(df)
        dfList = df.tolist()
        A = matlab.double(dfList)
        imfs = eng.iceemdan(A)
        imfs = np.array(imfs)
        return imfs

    """
    绘制向量
    :param data:分解后的分量
    """
    def plotimfs(imfs, data):
        plt.figure(figsize=(12, 18))
        plt.subplots_adjust(hspace=0.5)
        plt.subplot(imfs.shape[0] + 3, 1, 1)
        plt.plot(data, 'r')
        plt.title("Signal Input")
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        for i in range(imfs.shape[0]):
            plt.subplot(imfs.shape[0] + 3, 1, i + 2)
            plt.plot(imfs[i], 'g')
            plt.ylabel("imf %i" % (i + 1), fontsize=15)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.grid(True)
