# PYTHON基础库
from typing import Optional, Callable, Union, get_origin, get_args  # 类型提示
from functools import wraps  # 函数对象操作
import inspect  # 函数检查
import copy # 对象复制

# 向量数值计算库
import numpy as np

# 高级数学分析库
from scipy import signal  # 信号处理包
from scipy import fft  # 快速傅里叶变换包
from scipy import stats  # 统计分析包
from scipy import interpolate  # 插值分析包

# 可视化绘图库
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Times New Roman"]  # 英文字体
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号
# 设置全局字体大小
plt.rcParams["font.size"] = 16  # 设置全局字体大小

from matplotlib import font_manager

zh_font = font_manager.FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf")  # 中文字体

FLOAT_EPS = np.finfo(float).eps  # 机器精度
PI = np.pi  # 圆周率