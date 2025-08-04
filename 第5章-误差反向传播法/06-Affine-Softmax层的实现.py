# Affine层
"""
神经网络的正向传播中进行的矩阵的乘积运算在几何学领域被称为“仿射变换”A。
因此，这里将进行仿射变换的处理实现为“Affine层”。

参考公式：
1. ∂L/∂X = ∂L/∂Y · Wᵀ   (损失函数对输入的梯度)
2. ∂L/∂W = Xᵀ · ∂L/∂Y   (损失函数对权重的梯度)

符号说明：
- L: 损失函数值（标量）
- X: 输入特征矩阵 [batch_size, input_dim]
- W: 权重矩阵 [input_dim, output_dim]
- Y: 前向传播输出矩阵 [batch_size, output_dim]
- ∂L/∂Y: 输出层梯度 [batch_size, output_dim]
- ᵀ 表示矩阵转置运算
- · 表示矩阵乘法

矩阵的乘积（“dot”节点）的反向传播可以通过组建使矩阵对应维度的元素个数一致的乘积运算而推导出来
"""

# 批版本的Affine层
"""
N个数据一起进行正向传播的情况，也就是批版本的Affine层。
"""
# 批版本的Affine层演示
import numpy as np

# 正向传播演示
x_dot_w = np.array([[0, 0, 0], [10, 10, 10]])  # 矩阵乘法结果
B = np.array([1, 2, 3])                       # 偏置向量
print(x_dot_w)                                 # 打印矩阵乘法结果
print(x_dot_w + B)                             # 打印加偏置后的结果

# 反向传播演示
dy = np.array([[1, 2, 3], [4, 5, 6]])          # 上游梯度
print(dy)
db = np.sum(dy, axis=0)                        # 偏置梯度：按样本维度求和
print(db)

class Affine:
    def __init__(self, W, b):
        """初始化仿射变换层
        Args:
            W: 权重矩阵 [input_dim, output_dim]
            b: 偏置向量 [output_dim]
        """
        self.W = W
        self.b = b
        self.X = None   # 缓存输入数据用于反向传播
        self.dW = None  # 权重梯度
        self.db = None  # 偏置梯度

    def forward(self, X):
        """正向传播：计算 Y = X·W + b
        Args:
            X: 输入数据 [batch_size, input_dim]
        Returns:
            输出数据 [batch_size, output_dim]
        """
        self.X = X  # 缓存输入数据用于反向传播
        out = np.dot(X, self.W) + self.b  # 矩阵乘法 + 广播偏置
        return out

    def backward(self, dout):
        """反向传播：计算梯度
        Args:
            dout: 上游梯度 [batch_size, output_dim]
        Returns:
            dX: 输入梯度 [batch_size, input_dim]
        """
        dX = np.dot(dout, self.W.T)         # 输入梯度：∂L/∂X = ∂L/∂Y·Wᵀ
        self.dW = np.dot(self.X.T, dout)     # 权重梯度：∂L/∂W = Xᵀ·∂L/∂Y
        self.db = np.sum(dout, axis=0)       # 偏置梯度：沿batch维度求和
        return dX



# Softmax-with-Loss层
"""
实现Softmax激活函数与交叉熵损失函数的组合层
包含正向传播（损失计算）和反向传播（梯度计算）
"""
from 共用资源.functions import *  # 导入softmax和交叉熵函数

class SoftmaxWithLoss:
    def __init__(self):
        """初始化损失层"""
        self.loss = None  # 损失值
        self.y = None     # softmax输出概率
        self.t = None     # 真实标签(one-hot编码)

    def forward(self, X, t):
        """正向传播：计算损失值
        Args:
            X: 输入数据 [batch_size, class_num]
            t: 真实标签 [batch_size, class_num] (one-hot)
        Returns:
            交叉熵损失值（标量）
        """
        self.t = t
        self.y = softmax(X)  # 计算softmax概率分布
        self.loss = cross_entropy_error(self.y, self.t)  # 计算交叉熵损失
        return self.loss

    def backward(self, dout=1):
        """反向传播：计算梯度
        Args:
            dout: 上游梯度（默认为1）
        Returns:
            dx: 输入梯度 [batch_size, class_num]
        """
        batch_size = self.t.shape[0]  # 获取批大小
        dx = (self.y - self.t) / batch_size  # 梯度公式：(预测值-真实值)/批大小
        return dx