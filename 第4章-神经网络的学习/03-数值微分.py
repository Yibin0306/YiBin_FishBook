import numpy as np
import matplotlib.pyplot as plt

# 数值微分（数值导数）的实现
# 不好的实现示例（存在精度问题）
def numerical_diff(f, x):
    h = 10e-50  # 过小的h值会导致舍入误差
    return (f(x + h) - f(x)) / h  # 前向差分近似

# 演示Python中的舍入误差问题
# 10e-50在32位浮点数中会被表示为0.0
print(np.float32(1e-50))  # 输出：0.0

# 改进的数值微分实现（中心差分法）
def numerical_diff(f, x):
    h = 1e-4  # 0.0001，经验证的最佳h值
    return (f(x + h) - f(x - h)) / (2 * h)  # 中心差分法，精度更高

# 示例函数：f(x) = 0.01x² + 0.1x
def function_1(x):
    return 0.01 * x**2 + 0.1 * x
# 绘制函数图像
x = np.arange(0, 20.0, 0.1)  # 创建从0到20，步长为0.1的数组
y = function_1(x)  # 计算函数值

plt.xlabel('x')
plt.ylabel('y')
plt.plot(x, y)  # 绘制函数曲线
plt.show()  # 显示图像

# 计算函数在x=5和x=10处的数值导数
print(numerical_diff(function_1, 5))   # x=5处的导数（理论值≈0.2）
print(numerical_diff(function_1, 10))  # x=10处的导数（理论值≈0.3）

# 偏导数示例（多变量函数）
def function_2(x):
    """二元函数：f(x₀, x₁) = x₀² + x₁²"""
    return x[0]**2 + x[1]**2  # 等价于np.sum(x**2)

# 计算点(3,4)处关于x₀的偏导数
def function_tmp1(x0):
    """固定x₁=4，创建关于x₀的一元函数"""
    return x0*x0 + 4.0**2.0  # f(x₀, 4) = x₀² + 16

# 在x₀=3处计算偏导数（理论值：2×3=6）
print(numerical_diff(function_tmp1, 3.0))

# 计算点(3,4)处关于x₁的偏导数
def function_tmp2(x1):
    """固定x₀=3，创建关于x₁的一元函数"""
    return 3.0**2.0 + x1*x1  # f(3, x₁) = 9 + x₁²

# 在x₁=4处计算偏导数（理论值：2×4=8）
print(numerical_diff(function_tmp2, 4.0))

"""
数值微分说明：
1. 数值微分用于近似计算导数，当解析解难以获得时
2. 中心差分法(f(x+h)-f(x-h))/(2h)比前向差分精度更高
3. h值选择很重要：太小导致舍入误差，太大导致截断误差
4. 偏导数计算：固定其他变量，仅变化目标变量

实际应用：
- 在神经网络中用于梯度检查（验证反向传播实现是否正确）
- 当函数形式复杂无法求解析导时提供近似梯度
"""