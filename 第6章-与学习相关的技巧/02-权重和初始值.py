"""
权重初始化实验：探索不同初始化方法对激活值分布的影响
实验目的：验证权重初始化对解决梯度消失问题的重要性
"""

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """Sigmoid激活函数"""
    return 1 / (1 + np.exp(-x))

# 实验参数设置
x = np.random.randn(1000, 100)  # 输入数据：1000个样本，每个样本100个特征
node_num = 100  # 每个隐藏层的神经元数量
hidden_layer_size = 5  # 隐藏层数量（5层深度网络）
activations = {}  # 存储每层激活值的字典

# 通过隐藏层进行前向传播
for i in range(hidden_layer_size):
    # 使用上一层的输出作为当前层的输入（第一层除外）
    if i != 0:
        x = activations[i - 1]  # 获取上一层的激活输出

    # === 权重初始化方法选择 ===
    # 方法1: 标准差为1的高斯分布（不推荐）
    # w = np.random.randn(node_num, node_num) * 1

    # 方法2: 标准差为0.01的高斯分布（过小会导致梯度消失）
    # w = np.random.randn(node_num, node_num) * 0.01

    # 方法3: Xavier初始化（推荐用于Sigmoid/Tanh）
    w = np.random.randn(node_num, node_num) / np.sqrt(node_num)

    # === 前向传播计算 ===
    z = np.dot(x, w)  # 线性变换：z = x·W
    a = sigmoid(z)  # 非线性激活：a = σ(z)
    activations[i] = a  # 存储当前层激活值

# === 可视化激活值分布 ===
for i, a in activations.items():
    plt.subplot(1, len(activations), i + 1)  # 创建子图
    plt.title(f"{i + 1}-layer")  # 设置标题（层数）
    # 绘制激活值直方图（范围0-1，30个柱）
    plt.hist(a.flatten(), 30, range=(0, 1))
plt.show()

"""
关键概念解释：
1. 权重初始化为0的问题：
   - 导致所有神经元学习相同的特征
   - 破坏网络的表达能力（对称权重问题）
   - 使深度网络无法有效训练

2. 梯度消失问题：
   - 深层网络中反向传播时梯度指数级减小
   - 导致浅层参数无法有效更新
   - 原因：激活函数导数值小于1时的连乘效应

3. 初始化方法选择：
   - Sigmoid/Tanh：推荐Xavier初始化（std=1/√n）
     * 保持各层激活值方差稳定
     * 防止激活值饱和（接近0或1）
   - ReLU：推荐He初始化（std=√(2/n)）
     * 补偿ReLU负区导数为0的特性
     * 保持正向传播中的信号强度

4. 实验观察点：
   - 标准差过大(std=1)：激活值集中0/1附近（梯度消失）
   - 标准差过小(std=0.01)：激活值集中0.5附近（信号衰减）
   - Xavier初始化：激活值保持良好分布（最佳实践）
"""

# === 权重初始化指南 ===
"""
激活函数       | 推荐初始化方法      | 数学公式
--------------|-------------------|-----------------------------
Sigmoid/Tanh | Xavier初始化      | W ∼ N(0, √(1/n_in))
ReLU/ELU     | He初始化          | W ∼ N(0, √(2/n_in))
Leaky ReLU   | He初始化变体       | W ∼ N(0, √(2/((1+α²)n_in)))
"""