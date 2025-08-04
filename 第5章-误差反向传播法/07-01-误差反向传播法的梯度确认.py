"""
数值微分的优点是实现简单，因此，一般情况下不太容易出错。而误差
反向传播法的实现很复杂，容易出错。所以，经常会比较数值微分的结果和
误差反向传播法的结果，以确认误差反向传播法的实现是否正确。确认数值
微分求出的梯度结果和误差反向传播法求出的结果是否一致（严格地讲，是
非常相近）的操作称为梯度确认（gradient check）。
"""

# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
from 数据.mnist import load_mnist
from 共用资源.two_layer_net import TwoLayerNet

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

# 求各个权重的绝对误差的平均值
for key in grad_numerical.keys():
    diff = np.average( np.abs(grad_backprop[key] - grad_numerical[key]) )
    print(key + ":" + str(diff))

"""
会输出如下结果。
 b1:9.70418809871e-13
 W2:8.41139039497e-13
 b2:1.1945999745e-10
 W1:2.2232446644e-13

从这个结果可以看出，通过数值微分和误差反向传播法求出的梯度的差
非常小。比如，第1层的偏置的误差是9.7e-13（0.00000000000097）。
这样一来，我们就知道了通过误差反向传播法求出的梯度是正确的，误差反向传播法的
实现没有错误。
"""