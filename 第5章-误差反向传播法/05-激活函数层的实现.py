# ReLU层
"""
y = {
    x      (x > 0)
    0      (x ≤ 0)
}
∂y/∂x = {
    1      (x > 0)
    0      (x ≤ 0)
}
如果正向传播时的输入x大于0，则反向传播会将上游的值原封不动地传给下游。
反过来，如果正向传播时的x小于等于0，则反向传播中传给下游的信号将停在此处。
"""

# ReLU层的实现
class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
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
Sigmoid函数:
  定义: h(x) = 1 / (1 + exp(-x))
"""






















