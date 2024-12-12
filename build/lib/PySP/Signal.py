"""
# Signal
xcw_package库的框架模块, 定义了一些基本的类, 实现xcw_package库其它模块的桥接

## 内容
    - class: 
        1. Signal: 自带时间、频率等采样信息的信号类
        2. Analysis: 信号分析基类, 用于创建其他复杂的信号分析、处理方法
    - function:
        1. resample: 信号任意时间段重采样函数
"""

from .dependencies import Optional
from .dependencies import np
from .dependencies import inspect
from .dependencies import wraps
from .dependencies import get_origin, get_args, Union
from .decorators import Check_Vars

from .Plot import plot_spectrum


# --------------------------------------------------------------------------------------------#
# --## ---------------------------------------------------------------------------------------#
# ------## -----------------------------------------------------------------------------------#
# ----------## -------------------------------------------------------------------------------#
class Signal:
    """
    自带采样信息的信号类, 可进行简单预处理操作

    参数:
    --------
    data : np.ndarray
        输入数据数组，用于构建信号
    label : str
        信号标签
    dt : float
        采样时间间隔
    or
    fs : int
        采样频率
    or
    T : float
        信号采样时长

    属性：
    --------
    data : np.ndarray
        输入信号的时序数据
    N : int
        信号长度
    dt : float
        采样时间间隔
    fs : int
        采样频率
    T : float
        信号采样时长
    df : float
        频率分辨率
    t_Axis : np.ndarray
        时间坐标序列
    f_Axis : np.ndarray
        频率坐标序列

    方法：
    --------
    info()
        输出信号的采样信息
    plot()
        绘制信号的时域图
    """

    @Check_Vars(
        {
            "data": {"ndim": 1},
            "label": {},
            "dt": {"OpenLow": 0},
            "fs": {"Low": 1},
            "T": {"OpenLow": 0},
        }
    )
    def __init__(
        self,
        data: np.ndarray,
        label: str,
        dt: Optional[float] = None,
        fs: Optional[int] = None,
        T: Optional[float] = None,
        t0: Optional[float] = 0,
    ):
        self.data = data
        self.N = len(data)
        # 只允许给出一个采样参数
        if not [dt, fs, T].count(None) == 2:
            raise ValueError("采样参数错误, 请只给出一个采样参数且符合格式要求")
        # -----------------------------------------------------------------------------------#
        # 采样参数初始化, dt, fs, T三者知一得三
        if dt is not None:
            self.dt = dt
            self.fs = 1 / dt
            self.df = self.fs / (self.N)  # 保证Fs=N*df
            self.T = self.N * self.dt  # 保证dt=T/N
        elif fs is not None:
            self.fs = fs
            self.dt = 1 / fs
            self.df = self.fs / (self.N)
            self.T = self.N * self.dt
        elif T is not None:
            self.T = T
            self.dt = T / self.N
            self.fs = 1 / self.dt
            self.df = self.fs / (self.N)
        else:
            raise ValueError("采样参数错误")
        self.t0 = t0
        # -----------------------------------------------------------------------------------#
        # 设置信号标签
        self.label = label

    # ---------------------------------------------------------------------------------------#
    @property
    def t_Axis(self) -> np.ndarray:
        """
        动态生成时间坐标轴
        """
        return (
            np.arange(0, self.N) * self.dt + self.t0
        )  # 时间坐标，t=[t0,t0+dt,t0+2dt,...,t0+(N-1)dt]

    # ---------------------------------------------------------------------------------------#
    @property
    def f_Axis(self) -> np.ndarray:
        """
        动态生成频率坐标轴
        """
        return np.linspace(
            0, self.fs, self.N, endpoint=False
        )  # 频率坐标，f=[0,df,2df,...,(N-1)df]

    # ---------------------------------------------------------------------------------------#
    def __array__(self) -> np.ndarray:
        """
        返回信号数据数组, 用于在传递给NumPy函数时自动调用
        """
        return self.data

    # ---------------------------------------------------------------------------------------#
    def info(self,print:bool=True) -> dict:
        """
        输出信号的采样信息

        参数:
        --------
        print : bool
            是否打印显示该信号采样信息, 默认为True

        返回:
        --------
        info_dict : dict
            信号的采样信息字典, 键为参数名, 值为含单位参数值字符串
        """
        info = (
            f"N: {self.N}\n"
            f"fs: {self.fs} Hz\n"
            f"t0: {self.t0:.3f} s\n"
            f"dt: {self.dt:.6f} s\n"
            f"T {self.T:.3f} s\n"
            f"t1: {self.t0+self.T:.3f} s\n"
            f"df: {self.df:.3f} Hz\n"
            f"fn: {self.fs / 2:.1f} Hz\n"
        )
        if print:
            print(f"{self.label}的采样参数: \n", info)
        # 将字符串转为字典
        info = [i.split(": ") for i in info.split("\n") if i]
        info_dict = {i[0]: i[-1] for i in info}
        return info_dict

    # ---------------------------------------------------------------------------------------#
    def plot(self, **kwargs) -> None:
        """
        绘制信号的时域波形图
        """
        title = kwargs.get("title", f"{self.label}时域波形图")
        kwargs.pop("title", None)
        xticks = kwargs.get("xticks", np.arange(self.t0, self.t0 + self.T, self.T / 10))
        plot_spectrum(
            self.t_Axis,
            self.data,
            xlabel="时间t/s",
            xticks=xticks,
            title=title,
            **kwargs,
        )


# --------------------------------------------------------------------------------------------#
class Analysis:
    @staticmethod
    def Plot(plot_type: str, plot_func: callable):
        def plot_decorator(func):
            def wrapper(self, *args, **kwargs):  # 针对Analysis类的方法进行装饰
                res = func(self, *args, **kwargs)
                if self.plot:
                    self.plot_kwargs["plot_save"] = self.plot_save
                    if plot_type == "1D":  # plot一维连线谱
                        Axis, data = res[0], res[1]
                        plot_func(Axis, data, **self.plot_kwargs)
                    elif plot_type == "2D":  # imshow二维热力谱图
                        Axis1, Axis2, data = res[0], res[1], res[2]
                        plot_func(Axis1, Axis2, data, **self.plot_kwargs)
                return res

            return wrapper

        return plot_decorator

    # ---------------------------------------------------------------------------------------#
    @staticmethod
    def Input(*var_checks):
        # 根据json输入生成对应的变量检查装饰器
        def decorator(func):
            @wraps(func)  # 保留原函数的元信息：函数名、参数列表、注释文档、模块信息等
            def wrapper(self, *args, **kwargs):  # 针对Analysis类的方法进行装饰
                # ---------------------------------------------------------------------------#
                # 获取函数输入变量
                Vars = inspect.signature(func)
                bound_args = Vars.bind(self, *args, **kwargs)
                bound_args.apply_defaults()
                # 获取变量的类型注解
                annotations = func.__annotations__
                var_checks_json = var_checks[0]
                # ---------------------------------------------------------------------------#
                # 按指定方式检查指定的变量
                for var_name in var_checks_json:
                    var_value = bound_args.arguments.get(var_name)  # 变量实际值
                    var_type = annotations.get(var_name)  # 变量预设类型
                    var_cond = var_checks_json[var_name]  # 变量额外检查条件
                    # -----------------------------------------------------------------------#
                    # 对于传值的函数参数进行类型检查
                    if var_value is not None:
                        # -------------------------------------------------------------------#
                        # 处理 Optional 类型
                        if get_origin(var_type) is Union:
                            var_type = get_args(var_type)[0]
                        # 处理float变量的int输入
                        if var_type is float and isinstance(var_value, int):
                            var_value = float(var_value)
                        # 检查输入值类型是否为预设类型
                        if var_type and not isinstance(var_value, var_type):
                            raise TypeError(
                                f"输入变量 '{var_name}' 类型不为要求的 {var_type.__name__}, 实际为 {type(var_value).__name__}"
                            )
                        # 针对某些变量类型进行额外检查
                        # -------------------------------------------------------------------#
                        # array类检查
                        if isinstance(var_value, np.ndarray):
                            # 条件1：数组维度检查
                            if "ndim" in var_cond:
                                if var_value.ndim != var_cond["ndim"]:
                                    raise ValueError(
                                        f"输入array数组 '{var_name}' 维度不为要求的 {var_cond['ndim']}, 实际为{var_value.ndim}"
                                    )
                        # -------------------------------------------------------------------#
                        # int类
                        if isinstance(var_value, int):
                            # 条件1：下界检查
                            if "Low" in var_cond:
                                if not (var_cond["Low"] <= var_value):
                                    raise ValueError(
                                        f"输入int变量 '{var_name}' 小于要求的下界 {var_cond["Low"]}, 实际为{var_value}"
                                    )
                            # 条件2：上界检查
                            if "High" in var_cond:
                                if not (var_value <= var_cond["High"]):
                                    raise ValueError(
                                        f"输入int变量 '{var_name}' 大于要求的上界 {var_cond["High"]}, 实际为{var_value}"
                                    )
                        # -------------------------------------------------------------------#
                        # float类
                        if isinstance(var_value, float):
                            # 条件1：闭下界检查
                            if "CloseLow" in var_cond:
                                if not (var_cond["CloseLow"] <= var_value):
                                    raise ValueError(
                                        f"输入float变量 '{var_name}' 小于要求的下界 {var_cond["CloseLow"]}, 实际为{var_value}"
                                    )
                            # 条件2：闭上界检查
                            if "CloseHigh" in var_cond:
                                if not (var_value <= var_cond["CloseHigh"]):
                                    raise ValueError(
                                        f"输入float变量 '{var_name}' 大于要求的上界 {var_cond["CloseHigh"]}, 实际为{var_value}"
                                    )
                            # 条件3：开下界检查
                            if "OpenLow" in var_cond:
                                if not (var_cond["OpenLow"] < var_value):
                                    raise ValueError(
                                        f"输入float变量 '{var_name}' 小于或等于要求的下界 {var_cond["OpenLow"]}, 实际为{var_value}"
                                    )
                            # 条件4：开上界检查
                            if "OpenHigh" in var_cond:
                                if not (var_value < var_cond["OpenHigh"]):
                                    raise ValueError(
                                        f"输入float变量 '{var_name}' 大于或等于要求的上界 {var_cond["OpenHigh"]}, 实际为{var_value}"
                                    )
                        # -------------------------------------------------------------------#
                        # str类
                        if isinstance(var_value, str):
                            # 条件1：字符串内容检查
                            if "Content" in var_cond:
                                if var_value not in var_cond["Content"]:
                                    raise ValueError(
                                        f"输入str变量 '{var_name}' 不在要求的范围 {var_cond['Content']}, 实际为{var_value}"
                                    )
                        # -------------------------------------------------------------------#
                        # Signal类
                        if isinstance(var_value, Signal):
                            pass
                # ---------------------------------------------------------------------------#
                return func(self, *args, **kwargs)  # 检查通过，执行类方法

            return wrapper

        return decorator

    # ---------------------------------------------------------------------------------------#
    def __init__(
        self, Sig: Signal, plot: bool = False, plot_save: bool = False, **kwargs
    ):
        self.Sig = Sig
        # 绘图参数全局设置
        self.plot = plot
        self.plot_save = plot_save
        self.plot_kwargs = kwargs


# --------------------------------------------------------------------------------------------#
@Check_Vars({"Sig":{},"down_fs": {"Low": 1}, "T": {"OpenLow": 0}})
def resample(
    Sig: Signal, down_fs: int, t0: float = 0, T: Optional[float] = None
) -> Signal:
    """
    对信号进行任意时间段的重采样

    参数:
    --------
    Sig : Signal
        输入信号
    down_fs : int
        重采样频率
    t0 : float
        重采样起始时间
    t1 : float
        重采样时间长度

    返回:
    --------
    resampled_Sig : Signal
        重采样后的信号
    """
    # 获取重采样间隔点数
    if down_fs > Sig.fs:
        raise ValueError("新采样频率应不大于原采样频率")
    else:
        ration = int(Sig.fs / down_fs)
    # 获取重采样起始点的索引
    if not Sig.t0 <= t0 < (Sig.T + Sig.t0):
        raise ValueError("起始时间不在信号时间范围内")
    else:
        start_n = int((t0 - Sig.t0) / Sig.dt)
    # 获取重采样点数
    if T is None:
        resample_N = -1
    elif T + t0 >= Sig.T + Sig.t0:
        raise ValueError("重采样时间长度超过信号时间范围")
    else:
        resample_N = int(T / (Sig.dt * ration))  # N = T/(dt*ration)
    # -----------------------------------------------------------------------------------#
    # 对信号进行重采样
    resampled_data = Sig.data[start_n::ration][:resample_N]  # 重采样
    resampled_Sig = Signal(
        resampled_data, label="重采样" + Sig.label, dt=ration * Sig.dt, t0=t0
    )  # 由于离散信号，实际采样率为fs/ration
    return resampled_Sig