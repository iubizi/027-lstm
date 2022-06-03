from random import random

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Bidirectional



# 定义大小
n_timesteps = 100

####################
# 渐进式显存
####################

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

####################
# 构造数据集
####################

# 创建序列分类实例
def get_sequence(n_timesteps):
    
	# 在 [0,1] 中创建一个随机数序列
	x = np.array([random() for _ in range(n_timesteps)])
	# 计算截止值以更改类别值
	limit = n_timesteps / 4.0
	# 确定累积序列中每个项目的类别结果
	y = np.array([0 if x < limit else 1 for x in np.cumsum(x)])
	# 重塑输入和输出数据以适合 LSTM
	x = x.reshape(1, n_timesteps, 1)
	y = y.reshape(1, n_timesteps, 1)
	
	return x, y

####################
# 可视化
####################

x, y = get_sequence(n_timesteps)

print('x.shape =', x.shape)
print('y.shape =', y.shape)
print()

####################
# 模型
####################

# 建立LSTM模型
model = Sequential()

model.add( Bidirectional(LSTM(1000, return_sequences=True), input_shape=(n_timesteps, 1)) )
model.add( Bidirectional(LSTM(1000, return_sequences=True)) )
model.add( TimeDistributed(Dense(1, activation='sigmoid')) )

from tensorflow.keras.optimizers import Adam

model.compile(
    loss = 'binary_crossentropy',
    optimizer = Adam(learning_rate=1e-5),
    metrics = ['accuracy'],
    )

# 训练LSTM
# 这个模式很好
for epoch in range(1000):
	# 生成新的随机序列
	x, y = get_sequence(n_timesteps)
	# 在这个序列上拟合一个时期的模型
	model.fit(x, y, epochs=1, batch_size=1, verbose=2)
	
# 测试LSTM
x, y = get_sequence(n_timesteps)
yhat = model.predict_classes(x, verbose=0)
for i in range(n_timesteps):
        print('Expected:', y[0, i], 'Predicted', yhat[0, i])
