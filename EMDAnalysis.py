from typing import Optional, Union
import numpy as np
import matplotlib.pyplot as plt
from xcw_package import Signal

from scipy.signal import argrelextrema  # 极值点查找
from scipy.interpolate import UnivariateSpline  # 多项式插值


class EMDAnalysis:
    def __init__(self, **kwargs) -> None:
        # IMF判定的不对称容忍度
        self.asy_toler = kwargs.get("asy_toler", 0.08)
        # 单次sifting提取IMF的最大迭代次数
        self.sifting_times = kwargs.get("sifting_times", 10)
        # EMD分解终止准则
        self.stoppage_criteria = kwargs.get("stoppage_criteria", "c1")
        # 默认的EMD分析方法
        self.method = kwargs.get("method", "emd")

    def __call__(self, **args) -> None:
        if self.method == "emd":
            return self.emd(**args)

    def emd(self, s: Signal, max_Dectimes: int = 5) -> np.ndarray:
        """
        EMD分解

        Parameters
        ----------
        s : Signal
            待分解的信号
        max_Dectimes : int, optional
            最大分解次数, by default 5

        Returns
        -------
        IMFs : np.ndarray
            分解出的IMF分量
        Residue : np.ndarray
            分解后的残余分量
        """
        # 初始化
        datatoDec = np.array(s.data)
        IMFs = []  # 存放已分解出的IMF分量
        Residue = datatoDec.copy()  # 存放EMD分解若干次后的残余分量

        # EMD循环分解信号残余分量
        for i_EMDdec in range(max_Dectimes):
            # 提取Residue中的IMF分量
            imf = self.extractIMF(datatoSift=Residue, max_iterations=self.sifting_times)
            if imf is None:
                break  # 若当前Residue无法提取IMF分量，则EMD分解结束
            else:
                IMFs.append(imf)
                Residue = Residue - imf  # 更新此次分解后的残余分量

            if self.__stoppage_criteria(datatoDec, Residue, self.stoppage_criteria):
                break  # 若满足终止EMD分解的准则，则EMD分解结束

        IMFs = np.array(IMFs)
        return IMFs, Residue

    def extractIMF(
        self,
        datatoSift: np.ndarray,
        max_iterations: int = 10,
        plot: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """
        提取IMF分量

        Parameters
        ----------
        datatoSift : np.ndarray
            待提取IMF分量的数据
        max_iter : int, optional
            最大迭代次数, by default 10

        Returns
        -------
        np.ndarray
            提取出的IMF分量
        """
        imf = datatoSift.copy()  # sifting迭代结果
        for n in range(max_iterations):
            res = self.isIMF(imf, kappa=self.asy_toler)
            if plot:
                figsize = kwargs.get("figsize", (12, 5))
                plt.figure(figsize=figsize)
                plt.plot(imf, label="IMF")
                plt.plot(res[1], label="Mean")
                plt.title(f"IMF提取结果，迭代次数：{n+1}")
                plt.legend()
                plt.show()
                print(res[2])
            if res[0] == True:
                return imf
            else:
                if res[2] == "极值点不足，无法提取IMF分量":
                    return None
                else:
                    imf = imf - res[1]  # 减去平均线作为下一次迭代结果

        return imf  # 最大迭代次数后仍未提取到IMF分量

    def isIMF(self, imf: np.ndarray, kappa: float = 0.1) -> tuple:
        """
        判断输入数据是否为IMF分量

        Parameters
        ----------
        imf : np.ndarray
            输入数据
        kappa : float, optional
            IMF不对称度阈值, by default 0.1

        Returns
        -------
        tuple
            ([bool]是否为IMF分量的标志, [np.ndarray]均值线, [str]不满足的条件)
        """
        N = len(imf)
        condition1, condition2, condition3 = False, True, False

        # 查找局部极值
        max_index, min_index = self.__search_localextrum(imf)

        if len(max_index) < 2 or len(min_index) < 2:
            return (False, imf, "极值点不足，无法提取IMF分量")

        # 判断是否存在骑波现象
        if np.all(imf[max_index] >= 0) and np.all(imf[min_index] <= 0):
            condition3 = True

        # 判断信号的零极点个数是否相等或相差1
        extrumNO = len(max_index) + len(min_index)
        zeroNO = self.__zero_crossing(imf)
        if np.abs(extrumNO - zeroNO) <= 1:
            condition2 = True

        # 添加首尾点,防止样条曲线的端点摆动
        if max_index[0] != 0:
            max_index = np.concatenate(([0], max_index))
        if max_index[-1] != N - 1:
            max_index = np.append(max_index, N - 1)
        if min_index[0] != 0:
            min_index = np.concatenate(([0], min_index))
        if min_index[-1] != N - 1:
            min_index = np.append(min_index, N - 1)

        # 三次样条插值
        max_spline = UnivariateSpline(max_index, imf[max_index], k=3, s=0)
        upper_envelop = max_spline(np.arange(N))  # 获得上包络线
        min_spline = UnivariateSpline(min_index, imf[min_index], k=3, s=0)
        lower_envelop = min_spline(np.arange(N))  # 获得下包络线
        mean = (upper_envelop + lower_envelop) / 2  # 计算均值线

        # 判断信号是否局部对称
        SD = np.std(mean) / np.std(imf)
        if SD < kappa:
            condition1 = True

        fault = ""
        if condition1 and condition2 and condition3:
            return (True, mean, fault)
        else:
            if condition1 is False:
                fault += "局部不对称度过高;"
            if condition2 is False:
                fault += "零极点个数相差大于1;"
            if condition3 is False:
                fault += "存在骑波现象;"
            return (False, mean, fault)

    def zero_crossing(
        self, data: np.ndarray, neibhors: int = 5, threshold: float = 1e-10
    ) -> int:
        _data = np.array(data)
        num = neibhors // 2  # 计算零点的邻域点数
        _data = _data[_data != 0]  # 删除直接为0的点
        zero_index = np.diff(np.sign(_data)) != 0  # 找到0点
        zero_index = np.arange(len(zero_index))[zero_index]  # 将0点区间的起点作为索引
        # 计算零点左侧标准差
        diff1 = np.array(
            [np.std(_data[i - num : i + 1]) if i - num >= 0 else 1 for i in zero_index]
        )
        # 计算零点右侧标准差
        diff2 = np.array(
            [
                np.std(_data[i : i + num + 1]) if i + num + 1 < len(_data) else 1
                for i in zero_index
            ]
        )
        # 零点左、右侧标准差均需大于阈值
        zero_index = zero_index[np.logical_and(diff1 > threshold, diff2 > threshold)]
        return len(zero_index)

    def stoppage_criteria(
        self, raw: np.ndarray, residue: np.ndarray, criteria: str
    ) -> bool:
        if criteria == "c1":
            if np.max(np.abs(residue)) < np.max(np.abs(raw)) * 1e-2:
                return True
            else:
                return False
        elif criteria == "c2":
            if np.std(residue) < 0.01 * np.std(raw):
                return True
            else:
                return False
        elif criteria == "c3":
            if np.std(residue) < 1e-6:
                return True
            else:
                return False
        else:
            raise ValueError("Invalid criteria")

    def search_localextrum(
        self, data: np.ndarray, neibhbors: int = 5, threshold: float = 1e-10
    ) -> np.ndarray:
        num = neibhbors // 2  # 计算局部极值的邻域点数
        # 查找局部极值
        max_index = argrelextrema(data, np.greater, order=num)[0]
        min_index = argrelextrema(data, np.less, order=num)[0]

        # 计算局部极值左侧标准差
        diff1 = np.array([np.std(data[i - num : i + 1]) for i in max_index])
        # 计算局部极值右侧标准差
        diff2 = np.array([np.std(data[i : i + num + 1]) for i in max_index])
        # 极值点左、右侧标准差均需大于阈值
        max_index = max_index[np.logical_and(diff1 > threshold, diff2 > threshold)]

        diff1 = np.array([np.std(data[i - num : i + 1]) for i in min_index])
        diff2 = np.array([np.std(data[i : i + num + 1]) for i in min_index])
        # 剔除不满足条件的极值点
        min_index = min_index[np.logical_and(diff1 > threshold, diff2 > threshold)]

        return max_index, min_index
<<<<<<< HEAD
    
    

    
=======
>>>>>>> 0f22d1e8d9d8e86c222918c104c106d833a07e46
