# 导入必要的库
import numpy as np  # 用于数值计算和生成数据
import matplotlib.pyplot as plt  # 用于数据可视化

# ========== 第一部分：绘制基础sin函数曲线 ==========
# 生成数据
x = np.arange(0, 6, 0.1)  # 创建从0到6（不含6），步长为0.1的数组
y = np.sin(x)             # 计算x数组中每个值的正弦值

# 绘制图形
plt.plot(x, y)  # 绘制x-y折线图
plt.show()      # 显示图形窗口

# ========== 第二部分：同时绘制sin和cos函数（含样式定制） ==========
# 重新生成数据（与第一部分相同）
x = np.arange(0, 6, 0.1)
y1 = np.sin(x)  # sin函数值
y2 = np.cos(x)  # cos函数值

# 绘制双曲线并添加样式
plt.plot(x, y1, label='sin(x)')  # 绘制实线，添加图例标签
plt.plot(x, y2, linestyle='--', label='cos(x)')  # 绘制虚线并添加标签

# 添加图表元素
plt.xlabel('x')  # 设置x轴标签
plt.ylabel('y')  # 设置y轴标签
plt.title('sin(x) & cos(x)')  # 设置图表标题
plt.legend()  # 显示图例

plt.show()  # 显示图形

# ========== 第三部分：图像显示示例 ==========
# 注意：路径可能需要根据实际文件位置调整
from matplotlib.image import imread  # 导入图像读取函数

try:
    img = imread('../数据/lena.png')  # 读取图像文件
    plt.imshow(img)  # 将图像数据转换为绘图对象
    plt.show()       # 显示图像窗口
except FileNotFoundError:
    print("图像文件未找到，请检查路径")  # 错误处理提示