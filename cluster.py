import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans, DBSCAN, Birch, OPTICS
from matplotlib import pyplot
import sklearn

import matplotlib.dates as mdates

font1 = {'family': 'Microsoft YaHei',
         'weight': 'normal',
         'size': 13}

# plt.style.use(['science','no-latex'])

exchanges = ["binance", "coinbase", "huobi", "kraken", "kucoin"]

def data_split(data, train_rate, seq_len, pre_len=1):
    time_len, n_feature = data.shape
    train_size = int(time_len * train_rate)
    print("time_len = {}".format(time_len))
    print("train_size = {}".format(train_size))
    print("test_size = {}".format(time_len-train_size))

    

    train_data = data[0:train_size]
    trainX, trainY, testX, testY = [], [], [], []
    for i in range(train_size-seq_len-pre_len):
        a = train_data[i: i + seq_len + pre_len]
        trainX.append(a[0: seq_len])
        trainY.append(a[seq_len: seq_len + pre_len, 0])
    for i in range(train_size-seq_len-pre_len, time_len-pre_len-seq_len):
        b = data[i: i+seq_len+pre_len,:]
        testX.append(b[0:seq_len])
        testY.append(b[seq_len:seq_len+pre_len, 0])
    trainX1 = np.array(trainX)
    trainY1 = np.array(trainY)
    testX1 = np.array(testX)
    testY1 = np.array(testY)
    return trainX1, trainY1, testX1, testY1

def cluster():
    # method = "K-Means"
    # method = "DBSCAN"
    # method = "Birch"
    method = "OPTICS"
    for i in range(len(exchanges)):
        exchange = exchanges[i]
        file = open('./exchange/feature/' + exchange + '_ft.csv')
        df = pd.read_csv(file)

        data = df.values
        time_len, n_features = data.shape
        train_rate = 0.8
        train_size = int(time_len * train_rate)
        seq_len = 10
        
        trainX, trainY, testX, testY = data_split(data, train_rate=train_rate, seq_len=seq_len)
        scaled_data = data.copy()

        df['transaction_amount_usd']
        tempx = df['transaction_amount_usd'].values
        x = []
        for j in range(train_size, time_len):
            x.append([j,tempx[j]])
        x = np.array(x)

       
        if method=="K-Means":
            model = KMeans(n_clusters=3)
        elif method=="DBSCAN":
            model = DBSCAN(eps=1000, min_samples=5)
        elif method=="Birch":
            model = Birch(threshold=0.5, n_clusters=3)
        elif method=="OPTICS":
            model = OPTICS()
        
        model.fit(x)
        
        if method=="DBSCAN" or method=="OPTICS":
            yhat = model.fit_predict(x)
        else:
            yhat = model.predict(x)
        
        clusters = unique(yhat)
        pyplot.figure(figsize=(6,4),dpi=300)
        
        for  cluster in clusters:
            row_ix = where(yhat==cluster)
            print(row_ix)
            pyplot.scatter(x[row_ix, 0], x[row_ix, 1])
            pyplot.xlabel("时间", font1)
            pyplot.ylabel("交易量（美元）", font1)
        pyplot.savefig("./exchange/figure/"+method+"/"+exchange+".png")
        pyplot.show()
        # elif method=="DBSCAN":
        #
        # elif method=="GMM":
        #
        # else:
        #     print("Default")

if __name__=='__main__':
    cluster()