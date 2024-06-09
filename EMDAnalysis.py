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
        # 查找零极点的邻域点数
        self.neibhors = kwargs.get("neibhors", 5)
        # 查找零点时的变化阈值
        self.zerothreshold = kwargs.get("zerothreshold", 1e-6)
        # 查找极值点的变化阈值
        self.extrumthreshold = kwargs.get("extrumthreshold", 1e-6)
        # Sifting过程上下包络是否使用首尾点
        self.End_envelop = kwargs.get("End_envelop", False)

    def __call__(
        self, s: Signal, max_Dectimes: int = 5, plot: bool = False, **kwargs
    ) -> np.ndarray:
        if self.method == "emd":
            return self.emd(s, max_Dectimes, plot, **kwargs)

    def emd(
        self, s: Signal, max_Dectimes: int = 5, plot: bool = False, **kwargs
    ) -> np.ndarray:
        """
        EMD分解

        Parameters
        ----------
        s : Signal
            待分解的信号
        max_Dectimes : int, optional
            最大分解次数, by default 5
        plot : bool, optional
            是否绘制分解过程, by default False

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
        for i in range(max_Dectimes):
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

        if plot:
            figsize = kwargs.get("figsize", (12, 5))
            for i, IMF in enumerate(IMFs):
                plt.figure(figsize=figsize)
                plt.plot(IMF)
                plt.title(f"IMF{i+1}")
                plt.show()
            plt.figure(figsize=figsize)
            plt.plot(Residue)
            plt.title("Residue")
            plt.show()

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
        max_iterations : int, optional
            最大迭代次数, by default 10
        plot : bool, optional
            是否绘制提取过程, by default False

        Returns
        -------
        np.ndarray
            提取出的IMF分量
        """
        imf = datatoSift.copy()  # sifting迭代结果
        for n in range(max_iterations):
            res = self.isIMF(imf, kappa=self.asy_toler, end=self.End_envelop)
            if plot:
                figsize = kwargs.get("figsize", (12, 5))
                plt.figure(figsize=figsize)
                plt.plot(imf, label="IMF")
                plt.plot(res[1], label="Mean")
                plt.title(f"第{n+1}次迭代")
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

    def isIMF(self, imf: np.ndarray, kappa: float = 0.1, end: bool = False) -> tuple:
        """
        判断输入数据是否为IMF分量

        Parameters
        ----------
        imf : np.ndarray
            输入数据
        kappa : float, optional
            IMF不对称度阈值, by default 0.1
        end : bool, optional
            信号首尾点是否参与包络, by default False

        Returns
        -------
        tuple
            ([bool]是否为IMF分量的标志, [np.ndarray]均值线, [str]不满足的条件)
        """
        N = len(imf)
        condition1, condition2, condition3 = False, False, False

        # 查找局部极值
        max_index, min_index = self.__search_localextrum(
            imf, neibhbors=self.neibhors, threshold=self.extrumthreshold
        )

        if len(max_index) < 2 or len(min_index) < 2:
            return (False, imf, "极值点不足，无法提取IMF分量")

        # 判断是否存在骑波现象
        if np.all(imf[max_index] >= 0) and np.all(imf[min_index] <= 0):
            condition3 = True

        # 判断信号的零极点个数是否相等或相差1
        extrumNO = len(max_index) + len(min_index)
        zeroNO = len(
            self.__search_zerocrossing(
                imf, neibhors=self.neibhors, threshold=self.zerothreshold
            )
        )
        if np.abs(extrumNO - zeroNO) <= 1:
            condition2 = True

        if end:
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

    def __search_zerocrossing(
        self, data: np.ndarray, neibhors: int = 5, threshold: float = 1e-6
    ) -> int:
        _data = np.array(data)
        num = neibhors // 2  # 计算零点的邻域点数
        _data[1:-1] = np.where(
            _data[1:-1] == 0, 1e-10, _data[1:-1]
        )  # 将直接零点替换为一个极小值，防止一个零点计入两个区间，首尾点不处理
        zero_index = np.diff(np.sign(_data)) != 0  # 寻找符号相异的相邻区间
        zero_index = np.append(zero_index, False)  # 整体前移，将区间起点作为零点
        zero_index = np.where(zero_index)[0]

        # 计算零点左侧标准差
        diff1 = np.array(
            [np.std(_data[i - num : i + 1]) if i - num >= 0 else 1 for i in zero_index]
        )
        # 计算零点右侧标准差
        diff2 = np.array(
            [
                np.std(_data[i : i + num + 1]) if i + num + 1 <= len(_data) else 1
                for i in zero_index
            ]
        )
        # 零点左、右侧标准差均需大于阈值
        zero_index = zero_index[np.logical_and(diff1 > threshold, diff2 > threshold)]
        return zero_index

    def __search_localextrum(
        self, data: np.ndarray, neibhbors: int = 5, threshold: float = 1e-6
    ) -> np.ndarray:
        num = neibhbors // 2  # 计算局部极值的邻域点数
        # 查找局部极值
        max_index = argrelextrema(data, np.greater, order=num)[0]
        min_index = argrelextrema(data, np.less, order=num)[0]

        # 去除微小抖动产生的极值点
        diff1 = np.array(
            [np.std(data[i - num : i + 1]) if i - num >= 0 else 1 for i in max_index]
        )  # 计算局部极值左侧标准差
        diff2 = np.array(
            [
                np.std(data[i : i + num + 1]) if i + num + 1 <= len(data) else 1
                for i in max_index
            ]
        )  # 计算局部极值右侧标准差
        max_index = max_index[np.logical_and(diff1 > threshold, diff2 > threshold)]
        #
        diff1 = np.array(
            [np.std(data[i - num : i + 1]) if i - num >= 0 else 1 for i in min_index]
        )
        diff2 = np.array(
            [
                np.std(data[i : i + num + 1]) if i + num + 1 <= len(data) else 1
                for i in min_index
            ]
        )
        min_index = min_index[np.logical_and(diff1 > threshold, diff2 > threshold)]

        return max_index, min_index

    def __stoppage_criteria(
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
