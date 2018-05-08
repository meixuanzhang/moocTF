# -*- coding: UTF-8 -*-

# 引入 Matplotlib 的分模块 PyPlot
import matplotlib.pyplot as plt

import numpy as np

# 创建数据，等分100份-2 到2之间
X = np.linspace(-2, 2, 100)
#y = 3 * X + 4
y1 = 3 * X +4
y2 = X **2

# 创建图像
#plt.plot(X,y)
plt.plot(X,y1)
plt.plot(X,y2)

# 显示图像
plt.show()

