__all__ = ["ENTROPY"]

import numpy as np
from math import factorial

"""
求多尺度排列熵
"""
class ENTROPY:
    def __init__(self,x):
        self.x = x

    """
    将一维时间序列，生成矩阵
    """
    def _embed(self,x, order=3, delay=1):
        """Time-delay embedding.

        Parameters
        ----------
        x : 1d-array, shape (n_times)
            Time series
        order : int
            Embedding dimension (order)
        delay : int
            Delay.

        Returns
        -------
        embedded : ndarray, shape (n_times - (order - 1) * delay, order)
            Embedded time-series.
        """
        N = len(x)
        Y = np.empty((order, N - (order - 1) * delay))
        for i in range(order):
            Y[i] = x[i * delay:i * delay + Y.shape[1]]
        return Y.T

    def permutation_entropy(self,time_series, order, delay, normalize=False):
        x = np.array(time_series)
        hashmult = np.power(order, np.arange(order))

        # _embed的作用是生成上图中重构后的矩阵
        # argsort的作用是对下标排序，排序的标准是值的大小
        sorted_idx = self._embed(x, order=order, delay=delay).argsort(kind='quicksort')

        # np.multiply 对应位置相乘  sum是求每一行的和
        # hashmult一定要保证三个一样的值顺序不同 按位乘起来后 每一行加起来 大小不同 类似赋一个权重
        hashval = (np.multiply(sorted_idx, hashmult)).sum(1)  # [21 21 11 19 11]

        # Return the counts
        _, c = np.unique(hashval, return_counts=True)  # 重小到大每个数字出现的次数

        p = np.true_divide(c, c.sum())  # [0.4 0.2 0.4]  2/5=0.4

        pe = -np.multiply(p, np.log2(p)).sum()  # 根据公式
        if normalize:  # 如果需要归一化
            pe /= np.log2(factorial(order))
        return pe

    def util_granulate_time_series(self,time_series, scale):
        """Extract coarse-grained time series

        Args:
            time_series: Time series
            scale: Scale factor

        Returns:
            Vector of coarse-grained time series with given scale factor
        """
        n = len(time_series)
        b = int(np.fix(n / scale))
        temp = np.reshape(time_series[0:b * scale], (b, scale))
        cts = np.mean(temp, axis=1)
        return cts

    def multiscale_permutation_entropy(self,time_series, order=3, delay=1, scale=3):
        """Calculate the Multiscale Permutation Entropy

        Args:
            time_series: Time series for analysis
            m: Order of permutation ENTROPY
            delay: Time delay
            scale: Scale factor

        Returns:
            Vector containing Multiscale Permutation Entropy

        Reference:
            [1] Francesco Carlo Morabito et al. Multivariate Multi-Scale Permutation Entropy for
                Complexity Analysis of Alzheimer’s Disease EEG. www.mdpi.com/1099-4300/14/7/1186
            [2] http://www.mathworks.com/matlabcentral/fileexchange/37288-multiscale-permutation-entropy-mpe/content/MPerm.m
        """
        mspe = []
        for i in range(scale):
            coarse_time_series = self.util_granulate_time_series(time_series, i + 1)
            pe = self.permutation_entropy(coarse_time_series, order=order, delay=delay, normalize=True)
            mspe.append(pe)
        return mspe