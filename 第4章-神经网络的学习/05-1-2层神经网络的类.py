# coding: utf-8
import sys, os

sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
from 共用资源.functions import *
from 共用资源.gradient import numerical_gradient


class TwoLayerNet:
    """
    进行初始化。参数从头开始依次表示输入层的神经元数、隐藏层的神经元数、输出层的神经元数
    """
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 初始化权重
        """
        保存神经网络的参数的字典型变量（实例变量）。
        params['W1']是第1层的权重，params['b1']是第1层的偏置。
        params['W2']是第2层的权重，params['b2']是第2层的偏置
        """
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    # 进行识别（推理）。参数x是图像数据
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    # 计算损失函数的值。参数x是图像数据，t是正确解标签（后面3个方法的参数也一样）
    # x:输入数据, t:监督数据
    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)

    # 计算识别精度
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # 计算权重参数的梯度
    # x:输入数据, t:监督数据
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        """
        保存梯度的字典型变量（numerical_gradient()方法的返回值）。
        grads['W1']是第1层权重的梯度，grads['b1']是第1层偏置的梯度。
        grads['W2']是第2层权重的梯度，grads['b2']是第2层偏置的梯度
        """
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    # 计算权重参数的梯度。numerical_gradient()的高速版，将在下一章实现
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads

net = TwoLayerNet(784, 10, 10)
W1 = net.params['W1'].shape # (784, 100)
b1 = net.params['b1'].shape # (100,)
W2 = net.params['W2'].shape # (100, 10)
b2 = net.params['b2'].shape # (10,)

x = np.random.randn(100, 784) # 伪输入数据(100笔)
y = net.predict(x)

x = np.random.randn(100, 784) # 伪输入数据(100笔)
t = np.random.randn(100, 10)  # 伪正确解标签(100笔)

grads = net.numerical_gradient(x, t)  # 计算梯度
W1 = grads['W1'].shape  # (784, 100)
b1 = grads['b1'].shape  # (100,)
W2 = grads['W2'].shape  # (100, 10)
b2 = grads['b2'].shape  # (10,)
