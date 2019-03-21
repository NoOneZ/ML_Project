import numpy as np
import time
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

#获取数据
data = pd.read_csv("msft_stockprices_dataset.csv")
data = data.drop(['Date'], axis=1)
X = data.drop(['Close Price'], axis=1)
Y = data["Close Price"]
data.head(3)

#分隔数据集到训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=int(time.time()))

#进行标准化处理 实例化两个标准化API
std_x = StandardScaler()
x_train = std_x.fit_transform(x_train)
x_test = std_x.transform(x_test)
#目标值
std_y = StandardScaler()
y_train = std_y.fit_transform(y_train.reshape(-1, 1))
y_test = std_y.transform(y_test.reshape(-1, 1))


#estimator预测
#梯度下降求解方式预测结果
lr = SGDRegressor()
lr.fit(x_train, y_train)
#回归系数
print(lr.coef_)

# 预测测试集的收盘价
y_predict = std_y.inverse_transform(lr.predict(x_test))
#y_predict = lr.predict(x_test)
print("测试集中每只股票的预测价格:", y_predict)
print("梯度下降的均方误差:", mean_squared_error(std_y.inverse_transform(y_test), y_predict))


