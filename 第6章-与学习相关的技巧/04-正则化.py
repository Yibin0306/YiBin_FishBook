# 过拟合
"""
发生过拟合的原因，主要有以下两个。
• 模型拥有大量参数、表现力强。
• 训练数据少。
"""

# 权值衰减
"""
权值衰减是一直以来经常被使用的一种抑制过拟合的方法。
该方法通过在学习的过程中对大的权重进行惩罚，来抑制过拟合。
"""
"""
L2范数相当于各个元素的平方和。
用数学式表示的话，假设有权重W=(w1,w2, ... ,wn)，则L2范数可用计算出来。
除了L2范数，还有L1范数、L∞范数等。L1范数是各个元素的绝对值之和，相当于|w1|+|w2|+...+|wn|。
L∞范数也称为Max范数，相当于各个元素的绝对值中最大的那一个。
L2范数、L1范数、L∞范数都可以用作正则化项，它们各有各的特点，不过这里我们要实现的是比较常用的L2范数。
"""

# Dropout
"""
Dropout是一种在学习的过程中随机删除神经元的方法。
正向传播时传递了信号的神经元，反向传播时按原样传递信号；
正向传播时没有传递信号的神经元，反向传播时信号将停在那里。
"""
import numpy as np
class Dropout:
    def __init__(self, dropout_ratio=0.5):
        """
        初始化Dropout层
        Args:
            dropout_ratio: 神经元丢弃比例，默认0.5（50%丢弃率）
        """
        self.dropout_ratio = dropout_ratio  # 丢弃率
        self.mask = None  # 存储二值掩码矩阵（0表示丢弃，1表示保留）

    def forward(self, x, train_flg=True):
        """
        前向传播
        Args:
            x: 输入数据（任意维度）
            train_flg: 训练模式标志（True=训练，False=测试）
        Returns:
            输出数据（应用Dropout后）
        """
        if train_flg:
            # 训练模式：生成随机掩码
            # 创建与x同形状的随机矩阵，大于dropout_ratio的位置设为1
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            # 应用掩码：随机丢弃神经元
            return x * self.mask
        else:
            # 测试模式：不丢弃神经元，但缩放输出
            # 缩放因子 = 1 - dropout_ratio (补偿训练时的丢弃)
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        """
        反向传播
        Args:
            dout: 上游传来的梯度
        Returns:
            应用相同掩码后的梯度
        """
        # 仅通过前向传播中保留的神经元传递梯度
        return dout * self.mask