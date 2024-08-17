# 用滤失系数来计算dP_dG-复杂裂缝
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from scipy.interpolate import interp1d
from tkinter import messagebox
from numpy.polynomial.polynomial import Polynomial


def calculate_cl(P, Pf1, Pf2, Pc1, Pc2, Pf3, Pc3, Pf4, Pc, CL1, CL2, CL3, CL4, omega):
    """
    计算修正后的滤失系数 CL

    参数:
    P (numpy array): 压力数组
    Pf1 (float): 第一阶段的压力上限
    Pf2 (float): 第二阶段的压力上限
    Pc1 (float): 第一个闭合压力
    Pc2 (float): 第二个闭合压力
    Pf3 (float): 第三个压力阈值
    Pc3 (float): 第三个闭合压力
    Pf4 (float): 第四阶段的压力上限
    Pc (float): 裂缝最终的闭合压力
    CL1 (float): 第一阶段的滤失系数
    CL2 (float): 第二阶段的滤失系数
    CL3 (float): 第三阶段的滤失系数
    CL4 (float): 第四阶段的滤失系数
    omega (float): 用于递归修正的自由变量

    返回:
    numpy array: 修正后的滤失系数数组
    """
    CL = np.zeros_like(P)

    # 条件1: P >= Pf1，CL 取最大值 CL1
    mask1 = P >= Pf1
    CL[mask1] = CL1

    # 条件2: (P >= Pc1) & (P < Pf1)，CL 从 CL1 渐变到 CL2
    mask2 = (P >= Pc1) & (P < Pf1)
    if np.any(mask2):
        numerator = np.exp(omega * (P[mask2] / Pf1)) - np.exp(omega * (Pc1 / Pf1))
        denominator = np.exp(omega) - np.exp(omega * (Pc1 / Pf1))
        CL[mask2] = (CL1 - CL2) * (numerator / denominator) + CL2

    # 条件3: (P >= Pf2) & (P < Pc1)，CL 取最大值 CL2
    mask3 = (P >= Pf2) & (P < Pc1)
    CL[mask3] = CL2

    # 条件4: (P >= Pc2) & (P < Pf2)，CL 从 CL2 渐变到 CL3
    mask4 = (P >= Pc2) & (P < Pf2)
    if np.any(mask4):
        numerator = np.exp(omega * (P[mask4] / Pf2)) - np.exp(omega * (Pc2 / Pf2))
        denominator = np.exp(omega) - np.exp(omega * (Pc2 / Pf2))
        CL[mask4] = (CL2 - CL3) * (numerator / denominator) + CL3

    # 条件5: (P >= Pc2) & (P < Pf3)，CL 取最大值 CL3
    mask5 = (P >=Pf3) & (P < Pc2)
    CL[mask5] = CL3

    # 条件6: (P >= Pc3) & (P < Pf3)，CL 从 CL3 渐变到 CL4
    mask6 = (P >= Pc3) & (P < Pf3)
    if np.any(mask6):
        numerator = np.exp(omega * (P[mask6] / Pf3)) - np.exp(omega * (Pc3 / Pf3))
        denominator = np.exp(omega) - np.exp(omega * (Pc3 / Pf3))
        CL[mask6] = (CL3 - CL4) * (numerator / denominator) + CL4

    # 条件7: (P >= Pc3) & (P < Pf4)，CL 取最小值 CL4
    mask7 = (P >= Pf4) & (P < Pc3)
    CL[mask7] = CL4

    # 条件8: (P >= Pc) & (P < Pf4)，CL 从 CL4 渐变到 0
    mask8 = (P >= Pc) & (P < Pf4)
    if np.any(mask8):
        numerator = np.exp(omega * (P[mask8] / Pf4)) - np.exp(omega * (Pc / Pf4))
        denominator = np.exp(omega) - np.exp(omega * (Pc / Pf4))
        CL[mask8] = (CL4 - 0) * (numerator / denominator) + 0

    return CL






def plot_cl_curves(pressure, Pf1, Pc1, Pf2, Pc2, Pf3, Pc3, Pf4, Pc, CL1, CL2, CL3, CL4, omegas):
    """
    绘制滤失系数随压降的变化曲线

    参数:
    pressure (numpy array): 压力数组
    Pf1 (float): 第一阶段的压力上限
    Pc1 (float): 第一个闭合压力
    Pf2 (float): 第二阶段的压力上限
    Pc2 (float): 第二个闭合压力
    Pf3 (float): 第三个阶段的压力上限
    Pc3 (float): 第三个闭合压力
    Pf4 (float): 第四阶段的压力上限
    Pc (float): 裂缝最终的闭合压力
    CL1 (float): 第一阶段的滤失系数
    CL2 (float): 第二阶段的滤失系数
    CL3 (float): 第三阶段的滤失系数
    CL4 (float): 第四阶段的滤失系数
    omegas (list of floats): 用于计算的不同omega值
    """
    plt.figure(figsize=(12, 8))

    # 将压力数组按降序排列
    pressure_sorted = np.sort(pressure)[::-1]

    # 计算滤失系数并绘图
    for omega in omegas:
        CL = calculate_cl(pressure_sorted, Pf1, Pc1, Pf2, Pc2, Pf3, Pc3, Pf4, Pc, CL1, CL2, CL3, CL4, omega)

        # 找到压力数组中小于 Pc 的部分
        valid_indices = pressure_sorted >= Pc

        # 绘制修正后的滤失系数曲线
        plt.plot(pressure_sorted[valid_indices], CL[valid_indices], label=f'omega={omega}')

    plt.xlabel('压力(MPa)')
    plt.ylabel('滤失系数 CL')
    plt.title('滤失系数随压降的变化曲线')
    plt.legend()
    plt.grid(True)
    plt.gca().invert_xaxis()  # 反转X轴，使其递减
    plt.show()





# 从Excel文件中读取井口压力数据
file_path = r"C:\Users\fy\Desktop\test2.xlsx"  # 更新文件路径
data = pd.read_excel(file_path)
time = data['Time']  # 时间列
pressure = data['Pressure']  # 压力列


# 绘制压力曲线与时间的图表
plt.figure(figsize=(12, 8))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体，确保标签显示汉字
plt.rcParams['axes.unicode_minus'] = False  # 设置正常显示负号
plt.plot(time, pressure, label='施工泵压')
plt.xlabel('时间(s)')
plt.ylabel('压力(MPa)')
plt.legend()
plt.title('施工曲线')
plt.show()

# 创建主窗口
root = tk.Tk()
root.title("输入参数")

# 创建标签和输入框
labels = ['停泵时间 (秒)', 'pf1 (MPa)', 'pc1 (MPa)', 'pf2 (MPa)', 'pc2 (MPa)', 'pf3 (MPa)', 'pc3 (MPa)',
          'pf4 (MPa)', 'pc (MPa)', 'CL1', 'CL2', 'CL3', 'CL4']

entries = {}
for i, label in enumerate(labels):
    tk.Label(root, text=label).grid(row=i, column=0, padx=10, pady=5)
    entry = tk.Entry(root)
    entry.grid(row=i, column=1, padx=10, pady=5)
    entries[label] = entry




def polynomial_smoothing(x, y, degree=3):
    """
    使用多项式拟合进行平滑

    参数:
    x (numpy array): x轴数据
    y (numpy array): y轴数据
    degree (int): 多项式的度数

    返回:
    numpy array: 拟合后的数据
    """
    p = Polynomial.fit(x, y, degree)
    return p(x)

def on_submit():
    try:
        stop_time = float(entries[labels[0]].get())
        Pf1 = float(entries[labels[1]].get())
        Pc1 = float(entries[labels[2]].get())
        Pf2 = float(entries[labels[3]].get())
        Pc2 = float(entries[labels[4]].get())
        Pf3 = float(entries[labels[5]].get())
        Pc3 = float(entries[labels[6]].get())
        Pf4 = float(entries[labels[7]].get())
        Pc = float(entries[labels[8]].get())
        CL1 = float(entries[labels[9]].get())
        CL2 = float(entries[labels[10]].get())
        CL3 = float(entries[labels[11]].get())
        CL4 = float(entries[labels[12]].get())

        # 找到停泵时间对应的压力值
        stop_pressure = pressure[time == stop_time].values[0] if stop_time in time.values else None

        if stop_pressure is not None:
            print(f"停泵时间 {stop_time} 秒对应的停泵压力值(ISIP)是 {stop_pressure} MPa")
        else:
            print("停泵时间不在数据范围内，请检查输入的时间。")
            return

        # 截取停泵后的时间和压力数据
        time_after_stop = time[time >= stop_time].reset_index(drop=True)
        pressure_after_stop = pressure[time >= stop_time].reset_index(drop=True)

        # 检查 Pc 是否在合理范围内
        if Pc < pressure_after_stop.min()-1:
            tk.messagebox.showerror("输入错误", "输入的裂缝最终闭合压力 (Pc) 小于数据中的最小压力值。请重新输入。")
            return

        # 计算无因次时间 theta
        theta = time_after_stop / stop_time

        # 计算限制性G函数
        def G_00(theta):
            return 8 / np.pi * (np.sqrt(theta) - 1)

        def G_12(theta):
            return 4 / np.pi * (theta * np.arcsin(1 / np.sqrt(theta)) + np.sqrt(theta - 1) - np.pi / 2)

        def G_10(theta):
            return 16 / (3 * np.pi) * (theta ** (3/2) - (theta - 1) ** (3/2) - 1)

        G_00_values = G_00(theta)
        G_12_values = G_12(theta)
        G_10_values = G_10(theta)

        # 插值处理
        pressure_interp = np.linspace(pressure_after_stop.min(), pressure_after_stop.max(), num=1000)
        time_interp = interp1d(pressure_after_stop, time_after_stop, kind='linear')(pressure_interp)

        # 绘制滤失系数随压降的变化图
        omegas = [-100, -10, -1, 1, 10]
        plot_cl_curves(pressure_interp, Pf1, Pc1, Pf2, Pc2, Pf3, Pc3, Pf4, Pc, CL1, CL2, CL3, CL4, omegas)

        def compute_dP_dG(stop_pressure_bottom):
            # 初始化 CL_ratio 数组
            CL_ratio = np.zeros_like(pressure_after_stop)

            # 计算滤失系数与停泵时刻滤失系数的比值
            CL_values = calculate_cl(pressure_after_stop, Pf1, Pc1, Pf2, Pc2, Pf3, Pc3, Pf4, Pc, CL1, CL2, CL3, CL4,
                                     omega=10)

            for i, p in enumerate(pressure_after_stop):
                if p >= Pc1:
                    CL_ratio[i] = CL_values[i] / CL1
                elif p >= Pc2 and p < Pc1:
                    CL_ratio[i] = CL_values[i] / CL2
                elif p >= Pc3 and p < Pc2:
                    CL_ratio[i] = CL_values[i] / CL3
                elif p >= Pc and p < Pc3:
                    CL_ratio[i] = CL_values[i] / CL4

            # 将比值带入 dP/dG 公式
            dP_dG = stop_pressure_bottom * (7 / 6) * CL_ratio
            return dP_dG

        dP_dG = compute_dP_dG(pressure_after_stop)
        GdP_dG_00 = dP_dG * G_00(theta)

        # 使用多项式拟合进行平滑
        GdP_dG_00_smooth = polynomial_smoothing(theta, GdP_dG_00, degree=50)
        theta_smooth = np.linspace(theta.min(), theta.max(), len(GdP_dG_00_smooth))

        # 绘制图表
        plt.figure(figsize=(12, 8))
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体，确保标签显示汉字
        plt.rcParams['axes.unicode_minus'] = False  # 设置正常显示负号

        # 绘制G函数随时间变化的图
        plt.subplot(2, 1, 1)
        plt.plot(theta, G_00_values, label='G(0,0,θ)')
        plt.plot(theta, G_12_values, label='G(1/2,0,θ)')
        plt.plot(theta, G_10_values, label='G(1,0,θ)')
        plt.xlabel('无因次时间(θ)')
        plt.ylabel('G函数')
        plt.legend()
        plt.title('G函数曲线')

        # 绘制压力曲线和GdP/dG曲线
        ax1 = plt.subplot(2, 1, 2)
        color = 'tab:blue'
        ax1.set_xlabel('无因次时间(θ)')
        ax1.set_ylabel('压力 (MPa)', color=color)
        ax1.plot(theta, pressure_after_stop, label='Pressure', color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        # 创建第二个 y 轴
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('GdP/dG', color=color)
        ax2.plot(theta_smooth, GdP_dG_00_smooth, label='GdP/dG (0,0,θ) - 平滑', color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        # 添加图例
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        # 设置图表标题
        plt.title('压力曲线与GdP/dG曲线')

        # 调整布局
        plt.tight_layout()

        # 显示图表
        plt.show()

    except ValueError:
        print("请输入有效的数值。")



# 创建提交按钮
submit_button = ttk.Button(root, text="提交", command=on_submit)
submit_button.grid(row=len(labels), columnspan=2, pady=10)

# 运行主循环
root.mainloop()
