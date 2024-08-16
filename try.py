# # 用压差函数来计算dP_dG
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import tkinter as tk
# from tkinter import simpledialog
# from main import calculate_bottomhole_pressure_from_excel
# # 从Excel文件中读取井口压力数据
# file_path = r"C:\Users\fy\Desktop\test2.xlsx"  # 更新文件路径
# data = pd.read_excel(file_path)
# time = data['Time']  # 时间列
# pressure = data['Pressure']  # 压力列
#
# # 绘制压力曲线与时间的图表
# plt.figure(figsize=(12, 8))
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体，确保标签显示汉字
# plt.rcParams['axes.unicode_minus'] = False  # 设置正常显示负号
# plt.plot(time, pressure, label='施工泵压')
# plt.xlabel('时间(s)')
# plt.ylabel('压力(MPa)')
# plt.legend()
# plt.title('施工曲线')
# plt.show()
# calculate_bottomhole_pressure_from_excel(r"C:\Users\fy\Desktop\test.xlsx")
# # 创建主窗口
# root = tk.Tk()
# root.withdraw()  # 隐藏主窗口
#
# # 从用户获取停泵时间
# stop_time = simpledialog.askfloat("Input", "请输入停泵时间 (秒):", minvalue=0)
# # 找到停泵时间对应的压力值
# stop_pressure = pressure[time == stop_time].values[0] if stop_time in time.values else None
#
# if stop_pressure is not None:
#     print(f"停泵时间 {stop_time} 秒对应的压力值是 {stop_pressure} MPa")
# else:
#     print("停泵时间不在数据范围内，请检查输入的时间。")
#
# # 截取停泵后的时间和压力数据
# time_after_stop = time[time >= stop_time].reset_index(drop=True)
# pressure_after_stop = pressure[time >= stop_time].reset_index(drop=True)
#
# # 计算无因次时间 theta
# theta = time_after_stop / stop_time
#
# # 计算限制性G函数
# def G_00(theta):
#     return 8 / np.pi * (np.sqrt(theta) - 1)
#
# def G_12(theta):
#     return 4 / np.pi * (theta * np.arcsin(1 / np.sqrt(theta)) + np.sqrt(theta - 1) - np.pi / 2)
#
# def G_10(theta):
#     return 16 / (3 * np.pi) * (theta ** (3/2) - (theta - 1) ** (3/2) - 1)
#
# G_00_values = G_00(theta)
# G_12_values = G_12(theta)
# G_10_values = G_10(theta)
#
# # 从用户获取地层压力
# constant = simpledialog.askfloat("Input", "请输入地层压力 (MPa):", minvalue=0)
# # 获取停泵时刻的井底净压力
# stop_pressure_bottom = simpledialog.askfloat("Input", "请输入停泵时刻井底净压力 (MPa):", minvalue=0)
#
# # 计算dP/dG
# def compute_dP_dG(pressure_after_stop, stop_pressure, stop_pressure_bottom, constant):
#     dP_dG = stop_pressure_bottom * (7/6) * (((pressure_after_stop - constant) / (stop_pressure - constant)) ** (1/2))
#     return dP_dG
#
# dP_dG = compute_dP_dG(pressure_after_stop, stop_pressure, stop_pressure_bottom, constant)
#
# GdP_dG_00 = dP_dG * G_00(theta)
#
# # 绘制图表
# plt.figure(figsize=(12, 8))
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体，确保标签显示汉字
# plt.rcParams['axes.unicode_minus'] = False  # 设置正常显示负号
#
# # 绘制G函数随时间变化的图
# plt.subplot(2, 1, 1)
# plt.plot(theta, G_00_values, label='G(0,0,θ)')
# plt.plot(theta, G_12_values, label='G(1/2,0,θ)')
# plt.plot(theta, G_10_values, label='G(1,0,θ)')
# plt.xlabel('无因次时间(θ)')
# plt.ylabel('G函数')
# plt.legend()
# plt.title('G函数曲线')
#
# # 绘制压力曲线和GdP/dG曲线
# plt.subplot(2, 1, 2)
# plt.plot(theta, pressure_after_stop, label='Pressure')  # 使用theta作为横坐标
# plt.plot(theta, GdP_dG_00, label='GdP/dG (0,0,θ)')
# plt.xlabel('无因次时间(θ)')  # 更新横坐标标签
# plt.ylabel('压力和GdP/dG')
# plt.legend()
# plt.title('压力和GdP/dG-无因次时间(θ)')
# plt.tight_layout()
# plt.show()


# # 引入omega修正滤失系数
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import tkinter as tk
# from tkinter import simpledialog
# from tkinter import ttk
# from scipy.interpolate import interp1d
#
#
# def calculate_cl(P, Pf0, Pci, Pc, CL1, CL2, omega):
#     """
#     计算修正后的滤失系数 CL
#
#     参数:
#     P (numpy array): 压力数组
#     Pf0 (float): 天然裂缝开启压力
#     Pci (float): 只有人工裂缝参与滤失时的压力
#     Pc (float): 裂缝最终的闭合压力
#     CL1 (float): 只有人工裂缝参与滤失时的滤失系数
#     CL2 (float): 裂缝最终闭合时的滤失系数
#     omega (float): 自由变量
#
#     返回:
#     numpy array: 修正后的滤失系数数组
#     """
#     # 初始化滤失系数数组
#     CL = np.zeros_like(P)
#
#     # 创建掩码
#     mask1 = P > Pf0
#     mask2 = (P >= Pci) & (P <= Pf0)
#     mask3 = (P > Pc) & (P < Pci)
#
#     # 根据不同条件计算滤失系数
#     CL[mask1] = CL1
#     if np.any(mask2):
#         CL[mask2] = ((CL1 - CL2) * (((np.exp(omega * (P[mask2] - Pf0))) - (np.exp(omega * (Pci - Pf0)))) /
#                                     ((np.exp(omega)) - (np.exp(omega * (Pci - Pf0)))))) + CL2
#     CL[mask3] = CL2
#
#     return CL
#
#
# def plot_cl_curves(pressure_interp, Pf0, Pci, Pc, CL1, CL2, omegas):
#     plt.figure(figsize=(12, 8))
#     for omega in omegas:
#         CL = calculate_cl(pressure_interp, Pf0, Pci, Pc, CL1, CL2, omega)
#         plt.plot(pressure_interp, CL, label=f'omega={omega}')
#
#     # 画出未修正的折线
#     CL_step = np.piecewise(pressure_interp,
#                            [pressure_interp >= Pf0, (pressure_interp >= Pci) & (pressure_interp < Pf0),
#                             pressure_interp < Pci],
#                            [CL1, CL1, CL2])
#     plt.plot(pressure_interp, CL_step, 'k--', label='未修正的曲线')
#
#     plt.xlabel('压力 (MPa)')
#     plt.ylabel('滤失系数 (CL)')
#     plt.title('滤失系数随压降的变化图')
#     plt.legend()
#     plt.grid(True)
#     plt.show()
#
#
# # 从Excel文件中读取压力数据
# file_path = r"C:\Users\fy\Desktop\test2.xlsx"  # 更新文件路径
# data = pd.read_excel(file_path)
# time = data['Time']  # 时间列
# pressure = data['Pressure']  # 压力列
#
# # 绘制压力曲线与时间的图表
# plt.figure(figsize=(12, 8))
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体，确保标签显示汉字
# plt.rcParams['axes.unicode_minus'] = False  # 设置正常显示负号
# plt.plot(time, pressure, label='施工泵压')
# plt.xlabel('时间(s)')
# plt.ylabel('压力(MPa)')
# plt.legend()
# plt.title('施工曲线')
# plt.show()
#
# # 创建主窗口
# root = tk.Tk()
# root.title("输入参数")
#
# # 创建标签和输入框
# labels = ['停泵时间 (秒)', '天然裂缝开启压力 (MPa)', '只有人工裂缝参与滤失时的压力 (MPa)',
#           '裂缝最终的闭合压力 (MPa)', '只有人工裂缝参与滤失时的滤失系数',
#           '裂缝最终闭合时的滤失系数']
#
# entries = {}
# for i, label in enumerate(labels):
#     tk.Label(root, text=label).grid(row=i, column=0, padx=10, pady=5)
#     entry = tk.Entry(root)
#     entry.grid(row=i, column=1, padx=10, pady=5)
#     entries[label] = entry
#
#
# def on_submit():
#     try:
#         stop_time = float(entries[labels[0]].get())
#         Pf0 = float(entries[labels[1]].get())
#         Pci = float(entries[labels[2]].get())
#         Pc = float(entries[labels[3]].get())
#         CL1 = float(entries[labels[4]].get())
#         CL2 = float(entries[labels[5]].get())
#
#         # 找到停泵时间对应的压力值
#         stop_pressure = pressure[time == stop_time].values[0] if stop_time in time.values else None
#
#         if stop_pressure is not None:
#             print(f"停泵时间 {stop_time} 秒对应的压力值是 {stop_pressure} MPa")
#         else:
#             print("停泵时间不在数据范围内，请检查输入的时间。")
#             return
#
#         # 截取停泵后的时间和压力数据
#         time_after_stop = time[time >= stop_time].reset_index(drop=True)
#         pressure_after_stop = pressure[time >= stop_time].reset_index(drop=True)
#
#         # 插值处理
#         pressure_interp = np.linspace(pressure_after_stop.min(), pressure_after_stop.max(), num=1000)
#         time_interp = interp1d(pressure_after_stop, time_after_stop, kind='linear')(pressure_interp)
#
#         # 绘制滤失系数随压降的变化图
#         omegas = [-100, -10, -1, 1, 10]
#         plot_cl_curves(pressure_interp, Pf0, Pci, Pc, CL1, CL2, omegas)
#
#     except ValueError:
#         print("请输入有效的参数值。")
#
#
# submit_button = tk.Button(root, text="提交", command=on_submit)
# submit_button.grid(row=len(labels), column=0, columnspan=2, pady=10)
#
# root.mainloop()
#
