# -*- coding: UTF-8 -*-

# 引入 Matplotlib 的分模块 PyPlot
import matplotlib.pyplot as plt

import numpy as np

# 创建数据，等分100份-2 到2之间
X = np.linspace(-4, 4, 50)
#y = 3 * X + 4
y1 = 3 * X + 2
y2 = X **2


# 构建第一张图

plt.figure(num=1, figsize=(7,6))
plt.plot(X, y1)
plt.plot(X, y2, color="red", linewidth=3.0, linestyle="--")

# 构建第二张图
plt.figure(num=2)
plt.plot(X, y2, color="green")

# 显示图像
plt.show()

