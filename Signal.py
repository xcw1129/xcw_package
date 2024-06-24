import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 指定默认字体
plt.rcParams["axes.unicode_minus"] = False  # 解决保存图像是负号'-'显示为方块的问题

from scipy.signal import argrelextrema


class Signal:  # 信号类
    def __init__(self, data, dt=0, f_s=0, T=0):
        if not isinstance(data, np.ndarray):
            self.data = np.array(data)
        else:
            self.data = data
        self.N = len(data)
        if dt != 0:
            self.dt = dt
            self.f_s = 1 / dt
            self.T = (self.N - 1) * self.dt
        elif f_s != 0:
            self.f_s = f_s
            self.dt = 1 / f_s
            self.T = (self.N - 11) * self.dt
        elif T != 0:
            self.T = T
            self.dt = T / (self.N - 1)
            self.f_s = 1 / self.dt
        else:
            raise Exception("参数错误")

        self.df = self.f_s / (self.N - 1)  # 频率分辨率
        self.t_values = np.linspace(0, self.T, self.N)  # 时间坐标
        self.f_values = np.linspace(0, self.f_s, self.N)  # 频率坐标

    def info(self):
        info = (
            f"信号长度: {self.N}\n"
            f"采样频率: {self.f_s:.1f} Hz\n"
            f"采样间隔: {self.dt:.6f} s\n"
            f"信号采样时长: {self.T:.3f} s\n"
            f"频谱频率分辨率: {self.df:.3f} Hz\n"
            f"可分析频率上限: {self.f_s / 2:.1f} Hz\n"
        )
        print(info)

    def plot(self, **kwargs):  # 绘制信号的时域图
        plt.figure(figsize=(12, 5))
        plt.plot(self.t_values, self.data)
        title = kwargs.get("title", "时域波形")
        plt.title(title)
        plt.xlabel("时间t/s")
        xticks = kwargs.get("xticks", None)
        if xticks is not None:
            plt.xticks(xticks)
        plt.show()


def values_translate(values):  # 时间——频率坐标转换
    delta = values[1] - values[0]
    target = np.linspace(0, 1 / delta, len(values), endpoint=False)
    return target


def resample(s, new_f_s, new_N, start_t=0):  # 对信号进行重采样
    if type(new_N) != int:
        raise Exception("重采样点数应为整数")

    if new_f_s > s.f_s:
        raise Exception("新采样频率应小于原采样频率")
    ratio = int(s.f_s / new_f_s)  # 计算重采样点之间的间隔

    if start_t < 0 or start_t > s.T:
        raise Exception("起始时间不在信号范围内")
    start_index = int(start_t / s.dt)  # 计算起始点的索引

    new_data = s.data[start_index::ratio]
    if len(new_data) < new_N:
        raise Exception("新采样率过小，无法采集到足够多的点")
    new_data = new_data[:new_N]  # 截取前new_N个点
    new_s = Signal(
        new_data, dt=ratio * s.dt
    )  # 由于离散信号，目标重采样率与实际采样率有一定相差，故此处的dt为ratio*s.dt
    return new_s


def __smoothing(data, kernel):
    return np.convolve(data, kernel, mode="same")


def lowsmoothing(data, size, axis):
    s = lambda x: __smoothing(x, kernel=np.ones(size) / size)  # 定义平滑函数
    smoothing_data = np.apply_along_axis(s, axis, data)  # 平滑
    # 创建一个切片元组
    slice_tuple = [slice(None)] * data.ndim
    slice_tuple[axis] = slice(None, None, size)
    # 使用切片进行降采样
    downsampled_data = smoothing_data[tuple(slice_tuple)]
    return downsampled_data


def plot_findpeak(data, values, threshold, title=None):
    if len(data) != len(values):
        raise ValueError("data和values长度不一致")
    # 寻找峰值
    peak = argrelextrema(data, np.greater)
    peak_amplitude = data[peak]
    peak_values = values[peak]
    # 阈值筛选
    peak_values = peak_values[peak_amplitude > threshold]
    peak_amplitude = peak_amplitude[peak_amplitude > threshold]
    # 绘图指示峰值
    plt.figure(figsize=(12, 5))
    plt.plot(values, data)
    for val, amp in zip(peak_values, peak_amplitude):
        plt.annotate(
            f"{val:.1f}",
            (val, amp),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )
    plt.title(title)
    plt.show()


def plot_Spectrum(Axis: np.ndarray, data: np.ndarray, savefig=False, **kwargs):
    """
    根据轴和输入数据绘制单变量谱

    Parameters
    ----------
    Axis : np.ndarray
        横轴数据
    data : np.ndarray
        纵轴数据
    savefig : _type_
        是否保存svg图片

    Raises
    ------
    ValueError
        Axis和data的长度不一致
    """
    if len(Axis) != len(data):
        raise ValueError("Axis和data的长度不一致")

    figsize = kwargs.get("figsize", (12, 5))
    plt.figure(figsize=figsize)
    plt.plot(Axis, data)

    # 设置x轴标签
    xlabel = kwargs.get("xlabel", "时间t/s")
    plt.xlabel(xlabel)
    # 设置x轴范围
    xlim = kwargs.get("xlim", None)
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    # 设置y轴范围
    ylim = kwargs.get("ylim", None)
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    # 设置y轴显示方式
    yscale = kwargs.get("yscale", "linear")
    plt.yscale(yscale)
    # 设置标题
    title = kwargs.get("title", None)
    if title is not None:
        plt.title(title)

    if savefig:
        plt.savefig(title + "图", "svg")
    plt.show()
