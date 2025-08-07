"""
卷积神经网络(CNN)核心组件实现
包含卷积层和池化层的前向传播实现，使用im2col技巧加速运算
"""

import numpy as np

# =============================
# 测试数组维度理解
# =============================
# 创建4维随机数组：10个样本，1个通道，28x28像素
x = np.random.randn(10, 1, 28, 28)
print(x.shape)  # 输出: (10, 1, 28, 28) - (样本数, 通道数, 高度, 宽度)
print(x[0].shape)  # 输出: (1, 28, 28) - 第一个样本数据
print(x[1].shape)  # 输出: (1, 28, 28) - 第二个样本数据
print(x[0][0].shape)  # 输出: (28, 28) - 第一个样本的第一个通道

# =============================
# im2col技术说明
# =============================
"""
im2col函数将输入数据展开为适合滤波器(权重)计算的2维矩阵
实现原理：
1. 在输入数据的每个滤波器应用区域(三维方块)横向展开为1列
2. 对所有滤波器位置进行相同展开操作
3. 卷积运算转化为两个展开矩阵的乘积：
   - 输入数据展开为2D矩阵(列形式)
   - 滤波器纵向展开为1列
4. 计算结果需转换回4维输出格式(样本数, 通道数, 高, 宽)
"""

# 卷积层的实现
"""
im2col (input_data, filter_h, filter_w, stride=1, pad=0)
• input_data―由（数据量，通道，高，长）的 4维数组构成的输入数据
• filter_h―滤波器的高
• filter_w―滤波器的长
• stride―步幅
• pad―填充
"""
import sys , os
sys.path.append(os.pardir)
from 共用资源.util import im2col

# 测试im2col功能
# 单个样本: 1个样本，3通道，7x7输入
x1 = np.random.rand(1, 3, 7, 7)
# 使用5x5滤波器，步长1，无填充
col1 = im2col(x1, 5, 5, stride=1, pad=0)
print(col1.shape)  # 输出: (9, 75) - (窗口位置数, 5 * 5 * 3)

# 多个样本: 10个样本，3通道，7x7输入
x2 = np.random.rand(10, 3, 7, 7)
col2 = im2col(x2, 5, 5, stride=1, pad=0)
print(col2.shape)  # 输出: (90, 75) - (样本数*窗口位置数, 5 * 5 * 3)

# =============================
# 卷积层实现类
# =============================
class Convolution:
    """实现卷积层的前向传播"""

    def initialize(self, W, b, stride=1, pad=0):
        """
        初始化卷积层参数

        参数:
        W : 滤波器权重 (FN, C, FH, FW)
        b : 偏置向量 (FN,)
        stride : 卷积步长 (默认1)
        pad : 填充大小 (默认0)
        """
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        """
        卷积前向传播计算

        参数:
        x : 输入数据 (N, C, H, W)

        返回:
        out : 卷积结果 (N, FN, OH, OW)
        """
        # 获取滤波器维度
        FN, C, FH, FW = self.W.shape
        # 获取输入数据维度
        N, C, H, W = x.shape

        # 计算输出特征图尺寸
        out_h = int(1 + (H + 2 * self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2 * self.pad - FW) / self.stride)

        # 使用im2col展开输入数据
        col = im2col(x, FH, FW, stride=self.stride, pad=self.pad)

        # 展开滤波器为2D矩阵: (FN, C*FH*FW) -> 转置为(C*FH*FW, FN)
        col_w = self.W.reshape(FN, -1).T

        # 矩阵乘法计算卷积结果 + 偏置
        out = np.dot(col, col_w) + self.b

        # 调整输出形状并转置维度:
        # 1. 调整为(N, out_h, out_w, FN)
        # 2. 转置为(N, FN, out_h, out_w) 符合CNN标准格式
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        return out

# =============================
# 池化层实现类
# =============================
class Pooling:
    """实现池化层(最大池化)的前向传播"""

    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        """
        初始化池化层参数

        参数:
        pool_h : 池化窗口高度
        pool_w : 池化窗口宽度
        stride : 滑动步长 (默认1)
        pad : 填充大小 (默认0)
        """
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        """
        池化前向传播计算

        参数:
        x : 输入数据 (N, C, H, W)

        返回:
        out : 池化结果 (N, C, OH, OW)
        """
        # 获取输入数据维度
        N, C, H, W = x.shape

        # 计算输出特征图尺寸
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        # 步骤1: 使用im2col展开输入数据
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)

        # 调整形状: 每行表示一个池化窗口的所有元素
        col = col.reshape(-1, self.pool_h * self.pool_w)

        # 步骤2: 沿行方向取最大值 (最大池化)
        out = np.max(col, axis=1)

        # 步骤3: 转换输出形状
        # 1. 调整为(N, out_h, out_w, C)
        # 2. 转置为(N, C, out_h, out_w) 恢复通道维度位置
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        return out


"""
实现要点总结:
1. im2col技术将卷积/池化操作转换为矩阵乘法，显著提升计算效率
2. 卷积层核心步骤:
   - 输入数据im2col展开
   - 滤波器reshape为2D矩阵
   - 矩阵乘法 + 偏置
   - 结果reshape并调整维度顺序
3. 池化层核心步骤:
   - 输入数据im2col展开
   - 取每个窗口的最大值(最大池化)
   - 结果reshape并调整维度顺序
4. 维度转置(transpose)确保输出符合(N, C, H, W)标准格式
"""