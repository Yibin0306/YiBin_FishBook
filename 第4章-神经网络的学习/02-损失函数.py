"""
神经网络以某个指标为线索寻找最优权重参数。
神经网络的学习中所用的指标称为损失函数（loss function）。
这个损失函数可以使用任意函数，但一般用均方误差和交叉熵误差等。
"""

import numpy as np

# 均方误差
"""
均方误差（MSE）公式：
       1     n
MSE = --- *  Σ   (y_true_i - y_pred_i)²
       n     i=1

其中：
- n: 样本数量
- y_true_i: 第i个样本的真实值
- y_pred_i: 第i个样本的预测值
- Σ: 求和符号（从i=1到n求和）
"""

# 实现均方误差函数
def mean_squared_error(y_true, y_pred):
    """计算均方误差损失值"""
    return 0.5 * np.sum((y_true - y_pred) ** 2)  # 使用0.5系数简化梯度计算


# 测试用例：正确标签为数字2（one-hot编码索引2）
y_pred = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  # 模型预测值（one-hot格式）

# 情况1：模型预测正确（最高概率0.6在索引2）
y_true = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]  # 实际概率分布
x = mean_squared_error(np.array(y_true), np.array(y_pred))
print(x)  # 输出损失值（应较小）

# 情况2：模型预测错误（最高概率0.6在索引7）
y_true = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
x = mean_squared_error(np.array(y_true), np.array(y_pred))
print(x)  # 输出损失值（应较大）



# 交叉熵误差
"""
交叉熵误差公式（多分类通用形式）：
       n     C
CE = - Σ     Σ   y_{i,c} * log(p_{i,c})
      i=1   c=1

其中：
- n: 样本数量
- C: 类别数量
- y_{i,c}: 第i个样本在类别c上的真实标签（one-hot编码，0或1）
- p_{i,c}: 模型预测第i个样本属于类别c的概率（0~1之间）
- log: 自然对数（通常以e为底）

简化形式（当使用one-hot标签时）：
E = -Σ Tk * logYk
     k
这里，log表示自然对数（以e为底）。 yk是神经网络的输出，tk是正确解标签。
tk中只有正确解标签的索引为1，其他均为0（one-hot表示）。
"""

# 实现交叉熵误差函数
def cross_entropy_error(y, t):
    """计算交叉熵误差（带数值稳定性处理）"""
    delta = 1e-7  # 微小值，防止log(0)导致数值溢出
    return -np.sum(t * np.log(y + delta))

"""
函数内部在计算np.log时，加上了一个微小值delta。
这是因为，当出现np.log(0)时，np.log(0)会变为负无限大的-inf，导致计算错误。
"""

# 测试用例：正确标签为数字2（one-hot编码索引2）
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  # 真实标签（one-hot）

# 情况1：模型预测正确（最高概率0.6在索引2）
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]  # 预测概率
x = cross_entropy_error(np.array(y), np.array(t))
print(x)  # 输出损失值（应较小）

# 情况2：模型预测错误（最高概率0.6在索引7）
y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
x = cross_entropy_error(np.array(y), np.array(t))
print(x)  # 输出损失值（应较大）

# mini-batch学习
"""
        1         
CE = -  ——  Σ  Σ   Tnk * logYnk
        N   n  k

这里：
- N: batch大小（样本数量）
- tnk: 第n个样本的第k个元素的值（真实标签）
- ynk: 第n个样本的第k个元素的预测概率
式子虽然看起来有一些复杂，其实只是把求单个数据的损失函数的式扩大到了N份数据，不过最后还要除以N进行正规化。

我们从全部数据中选出一部分，作为全部数据的“近似”。
神经网络的学习也是从训练数据中选出一批数据（称为mini-batch,小批量），然后对每个mini-batch进行学习。
比如，从60000个训练数据中随机选择100笔，再用这100笔数据进行学习。这种学习方式称为mini-batch学习。
"""

"""
mini-batch的损失函数利用部分样本近似整体数据分布：
1. 从训练数据中随机选择小批量样本（mini-batch）
2. 用这些数据计算损失函数近似值
3. 通过平均损失进行正规化（除以batch_size）
"""

# 会从0到59999之间随机选择10个数字
print(np.random.choice(60000, 10))

"""
mini-batch的损失函数也是利用一部分样本数据来近似地计算整体。
也就是说，用随机选择的小批量数据（mini-batch）作为全体训练数据的近似值。
"""

# mini-batch版交叉熵误差实现（支持one-hot标签）
def cross_entropy_error(y, t):
    """支持批量数据的交叉熵误差计算"""
    if y.ndim == 1:  # 处理单样本输入
        t = t.reshape(1, t.size)  # 转为二维数组（batch_size=1）
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]  # 获取batch大小
    return -np.sum(t * np.log(y + 1e-7)) / batch_size  # 计算batch平均损失

# 当监督数据是标签形式（非one-hot表示，而是像“2”“ 7”这样的标签）时，交叉熵误差可通过如下代码实现
# mini-batch版交叉熵误差（支持数字标签格式）
def cross_entropy_error(y, t):
    """支持数字标签的交叉熵误差计算"""
    if y.ndim == 1:  # 处理单样本输入
        t = t.reshape(1, t.size)  # 转为二维数组
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]  # 获取batch大小
    # 使用高级索引提取每个样本真实标签对应的预测概率
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

# 为何要设定损失函数
"""
神经网络不能直接以识别精度为指标的原因：
1. 精度不可微：识别精度对参数变化不敏感，多数位置导数为0
2. 梯度消失：参数微小变化通常不会改变预测结果，导致无法梯度下降
3. 不连续性：精度值呈阶跃式变化，无法提供平滑优化路径

损失函数（如MSE/交叉熵）的特性：
1. 连续可微：处处存在非零梯度
2. 平滑性：参数微小变化会引起损失值的连续变化
3. 梯度信息：提供明确的参数优化方向

阶跃函数和sigmoid函数：阶跃函数的斜率在绝大多数地方都为0，而sigmoid函数的斜率（切线）不会为0。
得益于这个斜率不会为0的性质，神经网络的学习得以正确进行。
"""