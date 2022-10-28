from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Bidirectional
import numpy as np

class BiLSTM:
    def __init__(self,x):
        self.x = x

    """
    将值数组转换为数据集矩阵,look_back是步长
    """
    def create_dataset(self,dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            # X按照顺序取值
            dataX.append(a)
            # Y向后移动一位取值
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    """
    重构数据集
    """
    def reconstruct(self,data):
        # 数据重构为3D [samples, time steps, features]
        data = np.reshape(data, (data.shape[0], 1, data.shape[1]))
        return data

    """
    建模
    """
    def BiLSTM(self,look_back,trainX,trainY):
        model = Sequential()
        model.add(Bidirectional(LSTM(4, input_shape=(1, look_back))))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=2)
        return model



