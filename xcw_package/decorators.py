from .dependencies import np
from .dependencies import inspect
from .dependencies import wraps
from .dependencies import get_origin, get_args, Union


# --------------------------------------------------------------------------------------------#
# --## ---------------------------------------------------------------------------------------#
# ------## -----------------------------------------------------------------------------------#
# ----------## -------------------------------------------------------------------------------#
def Check_Vars(*var_checks):
    # 根据json输入生成对应的变量检查装饰器
    def decorator(func):
        @wraps(func)  # 保留原函数的元信息：函数名、参数列表、注释文档、模块信息等
        def wrapper(*args, **kwargs):
            from .Signal import Signal

            # -------------------------------------------------------------------------------#
            # 获取函数输入变量
            Vars = inspect.signature(func)
            bound_args = Vars.bind(*args, **kwargs)
            bound_args.apply_defaults()
            # 获取变量的类型注解
            annotations = func.__annotations__
            var_checks_json = var_checks[0]
            # -------------------------------------------------------------------------------#
            # 按指定方式检查指定的变量
            for var_name in var_checks_json:
                var_value = bound_args.arguments.get(var_name)  # 变量实际值
                var_type = annotations.get(var_name)  # 变量预设类型
                var_cond = var_checks_json[var_name]  # 变量额外检查条件
                if var_value is not None:
                    # -----------------------------------------------------------------------#
                    # 处理 Optional 类型
                    if get_origin(var_type) is Union:
                        var_type = get_args(var_type)[0]
                    # 处理float变量的int输入
                    if var_type == float and isinstance(var_value, int):
                        var_value = float(var_value)
                    # 检查输入值类型是否为预设类型
                    if var_type and not isinstance(var_value, var_type):
                        raise TypeError(
                            f"输入变量 '{var_name}' 类型不为要求的 {var_type.__name__}, 实际为 {type(var_value).__name__}"
                        )
                    # 针对某些变量类型进行额外检查
                    # -----------------------------------------------------------------------#
                    # array类检查
                    if isinstance(var_value, np.ndarray):
                        # 条件1：数组维度检查
                        if "ndim" in var_cond:
                            if var_value.ndim != var_cond["ndim"]:
                                raise ValueError(
                                    f"输入array数组 '{var_name}' 维度不为要求的 {var_cond['ndim']}, 实际为{var_value.ndim}"
                                )
                    # -----------------------------------------------------------------------#
                    # int类
                    if isinstance(var_value, int):
                        # 条件1：下界检查
                        if "Low" in var_cond:
                            if not (var_cond["Low"] <= var_value):
                                raise ValueError(
                                    f"输入int变量 '{var_name}' 小于要求的下界 {var_cond['limit']}, 实际为{var_value}"
                                )
                        # 条件2：上界检查
                        if "High" in var_cond:
                            if not (var_value <= var_cond["High"]):
                                raise ValueError(
                                    f"输入int变量 '{var_name}' 大于要求的上界 {var_cond['limit']}, 实际为{var_value}"
                                )
                    # -----------------------------------------------------------------------#
                    # float类
                    if isinstance(var_value, float):
                        # 条件1：闭下界检查
                        if "CloseLow" in var_cond:
                            if not (var_cond["CloseLow"] <= var_value):
                                raise ValueError(
                                    f"输入float变量 '{var_name}' 小于要求的下界 {var_cond['limit']}, 实际为{var_value}"
                                )
                        # 条件2：闭上界检查
                        if "CloseHigh" in var_cond:
                            if not (var_value <= var_cond["CloseHigh"]):
                                raise ValueError(
                                    f"输入float变量 '{var_name}' 大于要求的上界 {var_cond['limit']}, 实际为{var_value}"
                                )
                        # 条件3：开下界检查
                        if "OpenLow" in var_cond:
                            if not (var_cond["OpenLow"] < var_value):
                                raise ValueError(
                                    f"输入float变量 '{var_name}' 小于或等于要求的下界 {var_cond['limit']}, 实际为{var_value}"
                                )
                        # 条件4：开上界检查
                        if "OpenHigh" in var_cond:
                            if not (var_value < var_cond["OpenHigh"]):
                                raise ValueError(
                                    f"输入float变量 '{var_name}' 大于或等于要求的上界 {var_cond['limit']}, 实际为{var_value}"
                                )
                    # Signal类
                    if isinstance(var_value, Signal):
                        pass
            # -------------------------------------------------------------------------------#
            return func(*args, **kwargs)  # 检查通过，执行函数

        return wrapper

    return decorator


def Plot(plot_type: str, plot_func: callable):
    def plot_decorator(func):
        def wrapper(*args, **kwargs):
            res = func(*args, **kwargs)  # 执行函数取得绘图数据
            plot = kwargs.get("plot", False)
            if plot:
                if plot_type == "1D":
                    Axis, data = res[0], res[1]
                    kwargs.pop("data", None)  # 防止静态方法的输入参数data干扰
                    plot_func(Axis, data, **kwargs)
                elif plot_type == "2D":
                    Axis1, Axis2, data = res[0], res[1], res[2]
                    plot_func(Axis1, Axis2, data, **kwargs)
            return res

        return wrapper

    return plot_decorator
