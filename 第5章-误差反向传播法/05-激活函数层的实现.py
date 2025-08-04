"""
激活函数层实现：ReLU 和 Sigmoid
"""

# ReLU层
"""
ReLU（Rectified Linear Unit）激活函数定义：
正向传播：y = { x (当x>0), 0 (当x≤0) }
反向传播：∂y/∂x = { 1 (当x>0), 0 (当x≤0) }

特性：
- 正向传播时输入大于0则直接输出，否则输出0
- 反向传播时仅允许正输入区域的梯度通过
- 计算高效，缓解梯度消失问题
"""

class ReLU:
    """ReLU激活函数实现类"""

    def __init__(self):
        """初始化ReLU层，创建mask缓存区"""
        self.mask = None  # 用于存储正向传播时的掩码（布尔数组）

    def forward(self, x):
        """
        正向传播计算
        参数:
            x: 输入数据（NumPy数组）
        返回:
            out: ReLU激活后的输出
        """
        self.mask = (x <= 0)  # 记录输入≤0的位置（True表示需要屏蔽）
        out = x.copy()  # 创建输入数据的副本
        out[self.mask] = 0  # 将≤0的位置置零
        return out

    def backward(self, dout):
        """
        反向传播计算
        参数:
            dout: 上游传来的梯度
        返回:
            dx: 传递给下游的梯度
        """
        dout[self.mask] = 0  # 屏蔽原始输入≤0位置的梯度
        dx = dout  # 梯度直接传递（正区域梯度为1）
        return dx

"""
这个变量mask是由True/False构成的NumPy数组，它会把正向传播时的输入x的元素中小于等于0的地方保存为True，
其他地方（大于0的元素）保存为False。如下例所示，mask变量保存了由True/False构成的NumPy数组。
"""
import numpy as np
x = np.array([[1.0, -0.5],[-2.0, 3.0]])
print(x)
mask = (x <= 0)
print(mask)


# sigmoid层
"""
Sigmoid激活函数定义：
正向传播：y = 1 / (1 + exp(-x))
反向传播 ∂L/∂y ---> sigmoid ---> ∂L/∂y * y * (1-y)       (y = 1 / (1 + exp(-x)))

特性：
- 将输入压缩到(0,1)区间
- 反向传播使用正向输出计算梯度
- 存在梯度消失问题
"""

class Sigmoid:
    """Sigmoid激活函数实现类"""

    def __init__(self):
        """初始化Sigmoid层，创建输出缓存区"""
        self.out = None  # 用于缓存正向传播的输出结果

    def forward(self, x):
        """
        正向传播计算
        参数:
            x: 输入数据（NumPy数组）
        返回:
            out: Sigmoid激活后的输出
        """
        out = 1 / (1 + np.exp(-x))  # 计算sigmoid函数
        self.out = out  # 缓存输出结果用于反向传播
        return out

    def backward(self, dout):
        """
        反向传播计算
        参数:
            dout: 上游传来的梯度
        返回:
            dx: 传递给下游的梯度
        """
        # 使用链式法则计算梯度：dy/dx = y(1-y)
        dx = dout * self.out * (1.0 - self.out)
        return dx