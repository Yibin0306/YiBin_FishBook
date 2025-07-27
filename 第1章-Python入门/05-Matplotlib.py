# sin函数曲线
import numpy as np
import matplotlib.pyplot as plt

# 生成数据
x = np.arange(0, 6, 0.1) # 以0.1为单位生成0到6的数据
y = np.sin(x)

# 绘制图形
plt.plot(x, y)
plt.show()

# pyplot的功能
# 生成数据
x = np.arange(0, 6, 0.1)
y1 = np.sin(x)
y2 = np.cos(x)

# 绘制图像
plt.plot(x, y1, label='sin(x)')
plt.plot(x, y2, linestyle ='--', label='cos(x)') # 用虚线绘制
plt.xlabel('x') # x轴标签
plt.ylabel('y') # y轴标签
plt.title('sin(x) & cos(x)') # 标题
plt.legend()
plt.show()

# 显示图像
import matplotlib.pyplot as plt
from matplotlib.image import imread
img = imread('..\数据集\lena.png')
plt.imshow(img)
plt.show()