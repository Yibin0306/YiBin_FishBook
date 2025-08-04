"""
神经网络的学习的目的是找到使损失函数的值尽可能小的参数。
这是寻找最优参数的问题，解决这个问题的过程称为最优化（optimization）

使用参数的梯度，沿梯度方向更新参数，并重复这个步骤多次，从而逐渐靠
近最优参数，这个过程称为随机梯度下降法（stochastic gradient descent），简称SGD
"""

# SGD
"""
W ← W - η ∂L/∂W
把需要更新的权重参数记为W，把损失函数关于W的梯度记为∂L/∂W。
η表示学习率，实际上会取0.01或0.001这些事先决定好的值。
式子中的←表示用右边的值更新左边的值。
"""
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]

# SGD的缺点
"""
SGD的缺点是，如果函数的形状非均向（anisotropic），比如呈延伸状，搜索的路径就会非常低效。
SGD低效的根本原因是，梯度的方向并没有指向最小值的方向。
"""

# Momentum
