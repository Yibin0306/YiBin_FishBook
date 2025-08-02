import numpy as np

# 由全部变量的偏导数汇总而成的向量称为梯度（gradient）。梯度可以像下面这样来实现。
def numerical_gradient(f, x):
    h = 1e-2 # 0.0001
    grad = np.zeros_like(x) # 生成和x形状相同的数组

    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h)的计算
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h)的计算
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val # 还原值

    return grad