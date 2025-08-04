# 神经网络学习的全貌图
"""
前提
    神经网络存在合适的权重和偏置，调整权重和偏置以便拟合训练数据的过程称为“学习”。
    神经网络的学习分成下面4个步骤。
步骤1（mini-batch）
    从训练数据中随机选择一部分数据。
步骤2（计算梯度）
    计算损失函数关于各个权重参数的梯度。
步骤3（更新参数）
    将权重参数沿梯度方向进行微小更新。
步骤4（重复）
    重复步骤1、步骤2、步骤3。
"""

# 对应误差反向传播的神经网络的实现
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
from 共用资源.layers import *
from 共用资源.gradient import numerical_gradient
from collections import OrderedDict


class TwoLayerNet:
    """
    进行初始化。
    参数从头开始依次是输入层的神经元数、隐藏层的神经元数、输出层的神经元数、初始化权重时的高斯分布的规模
    """
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 初始化权重
        """
        保存神经网络的参数的字典型变量。
        params['W1']是第1层的权重，params['b1']是第1层的偏置。
        params['W2']是第2层的权重，params['b2']是第2层的偏置。
        """
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # 生成层
        """
        保存神经网络的层的有序字典型变量。
        以layers['Affine1']、layers['ReLu1']、layers['Affine2']的形式，
        通过有序字典保存各个层。
        """
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        """
        神经网络的最后一层。本例中为SoftmaxWithLoss层
        """
        self.lastLayer = SoftmaxWithLoss()

    # 进行识别（推理）。参数x是图像数据
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    # x:输入数据, t:监督数据
    # 计算损失函数的值。参数X是图像数据、t是正确解标签
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    # 计算识别精度
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # x:输入数据, t:监督数据
    # 通过数值微分计算关于权重参数的梯度（同上一章）
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    # 通过误差反向传播法计算关于权重参数的梯度
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 设定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads