"""
神经网络激活函数详解

核心概念：
1. 激活函数的作用：实现输入信号的转换，决定神经元是否被激活
2. 激活函数的演进：
   感知机 → 阶跃函数
   神经网络 → sigmoid/ReLU等非线性函数
3. 非线性激活函数的重要性：使神经网络能够学习复杂模式
"""

import numpy as np
import matplotlib.pyplot as plt

# ================= 阶跃函数 =================
"""
阶跃函数 (Step Function):
  定义: h(x) = { 1 if x > 0, 0 otherwise }
  特点: 
    - 感知机使用的激活函数
    - 输出为二元值 (0或1)
    - 非线性但有间断点
"""

# 阶跃函数实现 (支持NumPy数组)
def step_function(x):
    """阶跃函数实现"""
    return np.array(x > 0, dtype=np.int_)

# 测试阶跃函数
x_arr = np.array([-1.0, 1.0, 2.0])
print("阶跃函数测试:")
print("输入:", x_arr)
print("输出:", step_function(x_arr))  # [0 1 1]

# 绘制阶跃函数图像
plt.figure(figsize=(8, 5))
x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y, label='Step Function')
plt.title("Step Function")
plt.ylim(-0.1, 1.1)  # 设置y轴范围
plt.grid(True)
plt.legend()
plt.show()

# ================= Sigmoid函数 =================
"""
Sigmoid函数:
  定义: h(x) = 1 / (1 + exp(-x))
  特点:
    - 神经网络常用激活函数
    - 输出在0到1之间 (概率特性)
    - 平滑连续曲线
    - 非线性函数
    - 存在梯度消失问题
"""

def sigmoid(x):
    """Sigmoid函数实现"""
    return 1 / (1 + np.exp(-x))

# 测试Sigmoid函数
print("\nSigmoid函数测试:")
print("输入:", x_arr)
print("输出:", sigmoid(x_arr))  # [0.26894142, 0.73105858, 0.88079708]

# 绘制Sigmoid函数图像
plt.figure(figsize=(8, 5))
x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y, label='Sigmoid', color='red')
plt.title("Sigmoid Function")
plt.ylim(-0.1, 1.1)
plt.grid(True)
plt.legend()
plt.show()

# sigmoid函数与阶跃函数的比较
"""
sigmoid函数是一条平滑的曲线，输出随着输入发生连续性的变化。
而阶跃函数以0为界，输出发生急剧性的变化。sigmoid函数的平滑性对神经网络的学习具有重要意义。
感知机中神经元之间流动的是0或1的二元信号，而神经网络中流动的是连续的实数值信号
它们具有相似的形状。也就是说，当输入信号为重要信息时，阶跃函数和sigmoid函数都会输出较大的值；
当输入信号为不重要的信息时，两者都输出较小的值。
还有一个共同点是，不管输入信号有多小，或者有多大，输出信号的值都在0到1之间。
"""

# 非线性函数
"""
阶跃函数和sigmoid函数还有其他共同点，就是两者均为非线性函数。
线性函数是一条笔直的直线。而非线性函数，顾名思义，指的是不像线性函数那样呈现出一条直线的函数。
神经网络的激活函数必须使用非线性函数。
线性函数的问题在于，不管如何加深层数，总是存在与之等效的“无隐藏层的神经网络”
因此，为了发挥叠加层所带来的优势，激活函数必须使用非线性函数。
"""

# ================= ReLU函数 =================
"""
ReLU函数 (Rectified Linear Unit):
  定义: h(x) = max(0, x)
  特点:
    - 现代深度学习最常用的激活函数
    - 计算简单高效
    - 缓解梯度消失问题
    - 存在神经元"死亡"问题
    - 非线性函数
"""

def relu(x):
    """ReLU函数实现"""
    return np.maximum(0, x) # maximum函数会从输入的数值中选择较大的那个值进行输出

# 绘制ReLU函数图像
plt.figure(figsize=(8, 5))
x = np.arange(-6.0, 6.0, 0.1)
y = relu(x)
plt.plot(x, y, label='ReLU', color='green')
plt.title("ReLU Function")
plt.ylim(-1, 5)
plt.grid(True)
plt.legend()
plt.show()


# ================= 函数对比分析 =================
"""
激活函数比较:
| 特性           | 阶跃函数     | Sigmoid      | ReLU          |
|----------------|-------------|--------------|--------------|
| 输出范围       | {0, 1}      | (0, 1)       | [0, ∞)        |
| 连续性         | 不连续       | 连续         | 连续           |
| 可导性         | 不可导       | 可导         | 部分可导        |
| 梯度消失问题   | 无          | 严重         | 缓解            |
| 计算复杂度     | 低          | 中(含指数)   | 低              |
| 常用场景       | 感知机       | 早期神经网络 | 现代深度学习      |
"""

# ================= 非线性函数的重要性 =================
"""
非线性激活函数的必要性:
  1. 线性函数问题:
      h(x) = cx (线性函数)
      多层线性变换等价于单层变换: h(h(h(x))) = c³x

  2. 神经网络需要非线性:
      - 只有非线性激活函数才能赋予多层网络强大的表达能力
      - 非线性变换使网络能够学习复杂模式和决策边界
      - 线性激活函数的多层网络等同于单层网络

  3. 为什么选择这些特定非线性函数:
      - Sigmoid: 平滑可导，适合概率输出
      - ReLU: 计算高效，缓解梯度消失
      - 阶跃函数: 理论模型，实际应用有限
"""