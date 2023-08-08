import pandas as pd
import matplotlib.pyplot as plt

def plot_csv_with_two_y_axes(csv_file_path, column_A, column_B):
    # 读取CSV文件
    df = pd.read_csv(csv_file_path)
    
    # 提取列A和列B的数据并取浮点数部分
    y_data_A = df[column_A].apply(lambda x: float(x.split()[0]))
    y_data_B = df[column_B].apply(lambda x: float(x.split()[0]))
    
    # 创建图形和坐标轴对象
    fig, ax1 = plt.subplots()
    
    # 绘制第一个y轴的数据（列A）
    color = 'tab:red'
    ax1.set_xlabel('x')
    ax1.set_ylabel(column_A, color=color)
    ax1.plot(df.index, y_data_A, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    # 创建第二个y轴并绘制数据（列B）
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel(column_B, color=color)
    ax2.plot(df.index, y_data_B, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    # 自动调整y轴刻度
    plt.tight_layout()
    
    # 显示图形
    plt.show()

# 输入文件路径和列名
csv_file_path = "allzero_gemm_fp8.csv"
column_A = " power.draw.instant [W]"
column_B = " clocks.current.sm [MHz]"

# 调用绘图函数
plot_csv_with_two_y_axes(csv_file_path, column_A, column_B)
