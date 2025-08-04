"""
神经网络的学习的目的是找到使损失函数的值尽可能小的参数。
这是寻找最优参数的问题，解决这个问题的过程称为最优化（optimization）

使用参数的梯度，沿梯度方向更新参数，并重复这个步骤多次，从而逐渐靠
近最优参数，这个过程称为随机梯度下降法（stochastic gradient descent），简称SGD
"""

"""
SGD (随机梯度下降) 优化器
更新规则: W ← W - η ∂L/∂W
其中：
  W - 待更新的权重参数
  ∂L/∂W - 损失函数关于W的梯度
  η - 学习率
缺点：对于非均向(anisotropic)的函数形状，搜索路径低效
SGD低效的根本原因是，梯度的方向并没有指向最小值的方向。
"""

class SGD:
    def __init__(self, lr=0.01):
        """初始化学习率参数"""
        self.lr = lr  # 学习率(η)，默认值0.01

    def update(self, params, grads):
        """执行参数更新操作
        Args:
            params: 参数字典 {'权重名': 权重值}
            grads: 对应梯度字典 {'梯度名': 梯度值}
        """
        for key in params.keys():
            # SGD更新规则: W = W - η * ∂L/∂W
            params[key] -= self.lr * grads[key]


"""
Momentum (动量) 优化器
更新规则:
  v ← av - η ∂L/∂W
  W ← W + v
其中：
  v - 动量速度(物理意义)
  a - 动量系数(摩擦系数)
优点：解决SGD在非均向函数中的低效搜索问题
"""
import numpy as np
class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        """初始化优化器参数"""
        self.lr = lr  # 学习率(η)，默认值0.01
        self.momentum = momentum  # 动量系数(a)，默认值0.9
        self.v = None  # 速度变量缓存

    def update(self, params, grads):
        """执行参数更新操作"""
        # 初始化速度变量(第一次调用时)
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                # 创建与参数形状相同的零数组
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            # 更新速度: v = a*v - η*∂L/∂W
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            # 更新参数: W = W + v
            params[key] += self.v[key]

"""
AdaGrad (自适应学习率) 优化器
更新规则:
  h ← h + (∂L/∂W) ⊙ (∂L/∂W)
  W ← W - η * (1/√h) * (∂L/∂W)
特点：参数元素变动越大，学习率衰减越明显

变量h它保存了以前的所有梯度值的平方和（⊙ 表示对应矩阵元素的乘法）。
然后，在更新参数时，通过乘以，就可以调整学习的尺度。
这意味着，参数的元素中变动较大（被大幅更新）的元素的学习率将变小。
也就是说，可以按参数的元素进行学习率衰减，使变动大的参数的学习率逐渐减小。
"""
class AdaGrad:
    def __init__(self, lr=0.01):
        """初始化学习率参数"""
        self.lr = lr  # 基础学习率(η)
        self.h = None  # 历史梯度平方和缓存

    def update(self, params, grads):
        """执行参数更新操作"""
        # 初始化历史梯度缓存(第一次调用时)
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                # 创建与参数形状相同的零数组
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            # 累积梯度平方: h = h + (∂L/∂W)^2
            self.h[key] += grads[key] * grads[key]
            # 自适应更新: W = W - η*(1/√h)*∂L/∂W
            # 添加1e-7防止除零错误
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)

# Adam
"""
直观地讲，就是融合了Momentum和AdaGrad的方法。
通过组合前面两个方法的优点，有望实现参数空间的高效搜索。
"""