"""
梯度计算与梯度下降法实现
本文件演示了数值梯度计算、梯度下降算法及其在神经网络中的应用
"""

import numpy as np


# 偏导数示例（多变量函数）
def function_2(x):
    """
    二元函数：f(x₀, x₁) = x₀² + x₁²
    参数:
        x: 包含两个元素的数组 [x₀, x₁]
    返回:
        计算结果 x₀² + x₁²
    """
    return x[0] ** 2 + x[1] ** 2  # 等价于np.sum(x**2)


# 数值梯度计算
def numerical_gradient(f, x):
    """
    使用中心差分法计算函数在给定点的梯度
    参数:
        f: 目标函数
        x: 需要计算梯度的点（numpy数组）
    返回:
        grad: 梯度向量（与x同形状）
    """
    h = 1e-2  # 差分步长（0.01）
    grad = np.zeros_like(x)  # 生成和x形状相同的全零数组

    # 对每个维度分别计算偏导数
    for idx in range(x.size):
        tmp_val = x[idx]  # 保存原始值

        # 计算f(x+h)
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # 计算f(x-h)
        x[idx] = tmp_val - h
        fxh2 = f(x)

        # 使用中心差分公式计算偏导数
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val  # 恢复原始值

    return grad


# 梯度下降法实现
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    """
    使用梯度下降法寻找函数最小值
    参数:
        f: 目标函数
        init_x: 初始点
        lr: 学习率（默认0.01）
        step_num: 迭代次数（默认100）
    返回:
        x: 优化后的参数
    """
    x = init_x.copy()  # 创建初始点的副本

    # 进行指定次数的迭代
    for i in range(step_num):
        grad = numerical_gradient(f, x)  # 计算当前点的梯度
        x = x - lr * grad  # 沿负梯度方向更新参数

    return x


# 测试梯度计算
if __name__ == '__main__':
    # 在不同点计算梯度
    print("梯度测试结果:")
    test1 = numerical_gradient(function_2, np.array([3.0, 4.0]))  # [6, 8]
    test2 = numerical_gradient(function_2, np.array([0.0, 2.0]))  # [0, 4]
    test3 = numerical_gradient(function_2, np.array([3.0, 0.0]))  # [6, 0]
    print(f"(3,4)处梯度: {test1}")
    print(f"(0,2)处梯度: {test2}")
    print(f"(3,0)处梯度: {test3}")

    # 使用梯度下降寻找最小值
    print("\n梯度下降求最小值:")
    init_x = np.array([-3.0, 4.0])
    result = gradient_descent(function_2, init_x, lr=0.1, step_num=100)
    print(f"初始点: {init_x} -> 优化结果: {result}")

"""
梯度会指向各点处的函数值降低的方向。更严格地讲，梯度指示的方向是各点处的函数值减小最多的方向

通过不断地沿梯度方向前进，逐渐减小函数值的过程就是梯度法（gradient method）。

函数的极小值、最小值以及被称为鞍点（saddle point）的地方，梯度为0。
鞍点是从某个方向上看是极大值，从另一个方向上看则是极小值的点。

寻找最小值的梯度法称为梯度下降法（gradient descent method），
寻找最大值的梯度法称为梯度上升法（gradient ascent method）。
一般来说，神经网络（深度学习）中，梯度法主要是指梯度下降法。

使用数学式表达梯度法
x₀ = x₀ - η(∂f/∂x₀)
x₁ = x₁ - η(∂f/∂x₁)
η表示更新量，在神经网络的学习中，称为学习率（learning rate）。
学习率决定在一次学习中，应该学习多少，以及在多大程度上更新参数。
学习率过大的话，会发散成一个很大的值；反过来，学习率过小的话，基本上没怎么更新就结束了。
也就是说，设定合适的学习率是一个很重要的问题，像学习率这样的参数称为超参数。
"""

# ------------------------ 神经网络梯度示例 ------------------------

# 神经网络的梯度计算
import sys, os

sys.path.append(os.pardir)

# 从共用资源模块导入必要函数
from 共用资源.functions import softmax, cross_entropy_error
from 共用资源.gradient import numerical_gradient


class simpleNet:
    """简单的神经网络类（单层全连接）"""

    def __init__(self):
        """初始化权重矩阵（2x3）"""
        self.W = np.random.randn(2, 3)  # 用标准正态分布初始化权重

    def predict(self, x):
        """前向传播计算输出"""
        return np.dot(x, self.W)  # 矩阵乘法

    def loss(self, x, t):
        """计算损失函数（交叉熵误差）"""
        z = self.predict(x)  # 原始输出
        y = softmax(z)  # 概率输出（softmax归一化）
        loss = cross_entropy_error(y, t)  # 计算损失
        return loss


# 测试神经网络梯度
if __name__ == '__main__':
    print("\n神经网络梯度示例:")
    net = simpleNet()
    print(f"初始权重:\n{net.W}")

    # 输入样本
    x = np.array([0.6, 0.9])
    p = net.predict(x)
    print(f"预测输出: {p}")

    # 获取最大概率索引
    agm = np.argmax(p)
    print(f"预测类别: {agm}")

    # 正确标签（one-hot编码）
    t = np.array([0, 0, 1])
    los = net.loss(x, t)
    print(f"损失值: {los:.4f}")


    # 计算梯度
    def f(W):
        """包装函数用于计算损失"""
        return net.loss(x, t)


    # 计算损失函数关于权重的梯度
    dw = numerical_gradient(f, net.W)
    print(f"权重梯度:\n{dw}")