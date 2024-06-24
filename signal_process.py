import numpy as np

import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 指定默认字体
plt.rcParams["axes.unicode_minus"] = False  # 解决保存图像是负号'-'显示为方块的问题

from scipy.signal import hilbert
from scipy.signal import stft
from scipy.signal import butter, lfilter
from scipy.fft import fft
from scipy.stats import gaussian_kde
from scipy.stats import kurtosis

import pywt

from .Signal import plot_Spectrum


def FTspectrum(data: np.ndarray, fs: float, plot: bool = False, **kwargs) -> np.ndarray:
    """
    计算信号的归一化傅里叶变换频谱

    Parameters
    ----------
    data : np.ndarray
        输入信号
    fs : float
        采样率
    plot : bool, optional
        是否绘制0~fN的频谱, by default False

    Returns
    -------
    (np.ndarray,np.ndarray)
        频率轴,频谱数据
    """
    N = len(data)
    fre = np.arange(0, N) * fs / N
    FT = np.array(fft(data)) / N

    # 绘制频谱
    if plot:
        plot_Spectrum(fre[: N // 2], np.abs(data)[: N // 2], **kwargs)

    return fre, FT


def plot_pdf(s,plot_num=200,title=None,result=False):  # 绘制PDF图
    density = gaussian_kde(s.data)#核密度估计
    amplitude = np.linspace(min(s.data), max(s.data), plot_num)#幅值域采样密度
    pdf = density(amplitude)#概率密度函数采样
    plt.figure(figsize=(12,5))
    plt.plot(amplitude, pdf)
    plt.title(title)
    plt.show()
    if result:
        return pdf
    else:
        return None

def plot_autocorr(s,title=None,result=False):  # 绘制自相关图
    mean = np.mean(s.data)
    autocorr = np.correlate(
        s.data - mean, s.data - mean, mode="full"
    )  # 计算自相关，减去均值以忽略直流分量
    autocorr = autocorr[s.N - 1 :] / autocorr[s.N - 1]  # 除以信号总能量归一化，并只取右半部
    time = np.arange(len(autocorr)) * s.dt
    plt.figure(figsize=(12,5))
    plt.plot(time, autocorr)
    plt.xlabel("t/s")
    plt.title(title)
    plt.show()
    if result:
        return autocorr
    else:
        return None

def plot_PSD(s, nperseg,f_scope=0,title=None,result=False):
    num_segment = s.N // nperseg  # 根据段长确定分段数，多余数据舍去
    power = np.zeros((num_segment, nperseg))  # 功率谱存储矩阵，行数为分段数，列数为段长

    for i in range(num_segment):  # 分段计算功率谱
        data = s.data[i * nperseg : (i + 1) * nperseg]  # 取出段数据
        fft_data = fft(data)
        power[i, :] = np.abs(fft_data) ** 2/nperseg  # 将结构存入对应行数
    power_spectrum = np.average(power, axis=0)  # 计算多段谱平均

    frequence = np.linspace(0, s.f_s, nperseg)  # 频率范围
    if f_scope != 0:
        mask = (frequence >= f_scope[0]) & (frequence <= f_scope[1])
    plt.figure(figsize=(12,5))
    plt.plot(frequence[mask], power_spectrum[mask])  # 只绘制正频率部分
    plt.title(title)
    plt.xlabel('f/Hz')
    plt.show()
    print('分辨率:',frequence[1]-frequence[0])
    if result:
        return power_spectrum
    else:
        return None

def plot_movingwindow(s, window_length, statistic_func,title=None,result=False):
    if window_length > s.N:
        raise ValueError("窗长度大于信号长度，无法绘制移动平均窗")
    # 初始化一个空的列表来存储结果
    results = []

    # 滑动窗口遍历数据
    for i in range(len(s.data) - window_length + 1):
        # 获取当前窗口的数据
        window_data = s.data[i : i + window_length]
        # 计算并存储当前窗口的统计量
        stat_result = statistic_func(window_data)
        results.append(stat_result)

    # 绘制移动窗图
    plt.figure(figsize=(12,5))
    plt.plot(s.t_values[: len(s.data) - window_length + 1], results)
    plt.xlabel("t/s")
    plt.title(title)
    if result:
        return results
    else:
        return None


def HTenvelope(data: np.ndarray, fs: float, plot=False, **kwargs) -> np.ndarray:
    N = len(data)
    analyze = hilbert(data)
    magnitude = np.abs(analyze)  # 解析信号幅值，即原信号包络
    magnitude -= np.mean(magnitude)  # 去除直流分量
    FT = np.abs(fft(magnitude)) / N
    f_Axis = np.arange(0, N) * (fs / N)
    if plot:
        plot_Spectrum(f_Axis[: N // 2], FT[: N // 2], **kwargs)
    return FT


def plot_stft(s, nperseg, density=False,title=None,result=False):
    if nperseg > s.N:
        raise ValueError("段长大于信号长度，无法绘制STFT图")
    num_segment = s.N // nperseg  # 根据段长确定分段数，多余数据舍去
    fft_matrix = np.zeros((num_segment, nperseg))  # 频谱存储矩阵，行数为分段数，列数为段长
    for i in range(num_segment):  # 分段
        data = s.data[i * nperseg: (i + 1) * nperseg]  # 取出段数据
        fft_data = np.abs(fft(data)) / nperseg
        fft_matrix[i, :] = fft_data  # 将结构存入对应行数
    fft_matrix=fft_matrix[:,:fft_matrix.shape[1]//2]#只显示正频率
    plt.imshow(np.abs(fft_matrix.T), aspect='auto', origin='lower', cmap='jet', extent=[0,s.T,0,s.f_s//2])
    plt.title(title)
    plt.xlabel('t/s')
    plt.ylabel('f/Hz')
    plt.colorbar(label='Magnitude')
    plt.show()
    if result:
        return fft_matrix
    else:
        return None

def plot_Dwt(s,wavelet,result=False):
    data=s.data.copy()
    LPF,HPF=pywt.Wavelet(wavelet).filter_bank[:2]
    data=np.pad(data,len(LPF)//2,mode="symmetric")#首尾平拓展
    cA=np.convolve(data,LPF,mode="valid")[::2]#低通滤波和降采样
    cD=np.convolve(data,HPF,mode="valid")[::2]#高通滤波和降采样
    cA = cA[:-1]
    cD = cD[:-1]  #保持长度为原始信号的一半
    time=s.t_values[::2]#降采样后的时间序列
    plt.figure(figsize=(12,10))
    plt.subplot(2,1,1)
    plt.plot(time,cD)
    plt.title("细节")
    plt.subplot(2,1,2)
    plt.plot(time,cA)
    plt.title("近似")
    plt.show()
    if result:
        return cA,cD
    else:
        return None

def plot_Wavedec(s,wavelet,level,result=False):
    coeffs=[]#存放小波系数
    cA=s.data.copy()#初始化0层分解的cA
    for i in range(level):#进行level层分解
        cA,cD=pywt.dwt(cA,wavelet)#进行一层分解
        coeffs.append(cD)#频率由高到低的细节系数
    coeffs.append(cA)#存储cA
    time=np.linspace(0,s.T,10)
    frequency=np.array((s.f_s/4,s.f_s/2))#小波系数的大致频率范围
    plt.figure(figsize=(12,5*(level+2)))
    plt.subplot(level+2,1,1)
    plt.plot(s.t_values,s.data)
    plt.title("原始信号")
    for i in range(level):
        time=np.linspace(0,s.T,len(coeffs[i]),endpoint=False)
        plt.subplot(level+2,1,i+2)
        plt.plot(time,coeffs[i])
        plt.title("第{}层细节:{}~{}Hz".format(i+1,frequency[0],frequency[1]))
        frequency/=2
    time=np.linspace(0,s.T,len(coeffs[level]),endpoint=False)
    plt.subplot(level+2,1,level+2)
    plt.plot(time,coeffs[level])
    plt.title("第{}层近似:{}~{}Hz".format(level,0,frequency[1]))
    plt.show()

    if result:
        return coeffs
    else:
        return None

def plot_spectralkurtosis(s,nperseg_list,title=None,result=False):
    res=[]
    for nperseg in nperseg_list:
        f, t, fft_data = stft(
            s.data, fs=s.f_s, nperseg=nperseg
        )  # 计算STFT，得到矩阵fft_data
        fft_data=np.abs(fft_data) #计算复数模
        sprectral_kurtosis = np.zeros(fft_data.shape[0])
        for i in range(fft_data.shape[0]):
            sprectral_kurtosis[i] = kurtosis(fft_data[i,:])+1#计算每个频率点的时间平均峭度，+1使复高斯信号的峭度为0
        plt.figure(figsize=(12,5))
        plt.plot(f, sprectral_kurtosis)
        plt.title(title)
        plt.xlabel("f/Hz")
        plt.show()
        res.append(sprectral_kurtosis)
    if result:
        return res
    else:
        return None

def p_a(x,alpha,Fs):
    T = len(x) / Fs
    t = np.arange(0, T, 1/Fs)#时间离散化
    res=np.zeros(len(alpha),dtype=complex)
    for i,a in enumerate(alpha):
        res[i]=np.mean(x* np.exp(-1j * 2 * np.pi * a* t))#计算alpha循环频率处时间平均
    return res

def bandpass(data,center_f,bandwidth, fs,order=7):
    nyq = 0.5 * fs
    low = (center_f-bandwidth/2) / nyq
    high = (center_f+bandwidth/2)  / nyq
    b, a = butter(order, [low, high], btype='band')
    y=lfilter(b, a, data)
    return y

def plot_sqrenvelop(s,alpha,f_scope=0,title=None,result=False):
    data=np.square(s.data)
    P=p_a(data,alpha,s.f_s)
    magnitude=np.abs(P)
    if f_scope != 0:
        mask = (alpha >= f_scope[0]) & (alpha <= f_scope[1])
    plt.figure(figsize=(12,5))
    plt.plot(alpha[mask],magnitude[mask])
    plt.title(title)
    plt.xlabel("alpha/Hz")
    plt.show()

def plot_cycMspectral(s,alpha,ctr_f,bw,title=None,result=False):
    P=np.zeros((len(ctr_f),len(alpha)),dtype=float)
    for i,f in enumerate(ctr_f):
        bpdata=bandpass(s.data,f,bw,s.f_s)
        sqrdata=np.square(bpdata)
        P[i]=np.abs(p_a(sqrdata,alpha,s.f_s))
    P[:,0]=0
    X,Y=np.meshgrid(alpha,ctr_f)
    fig=plt.figure(figsize=(10,10))
    ax=fig.add_subplot(111,projection='3d')
    ax.plot_surface(X,Y,P,cmap='rainbow')
    ax.set_xlabel('循环频率/Hz')
    ax.set_ylabel('谱频率/Hz')
    plt.show()
