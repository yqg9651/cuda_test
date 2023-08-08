import csv
import matplotlib.pyplot as plt
import sys
import re

# 从命令行获取参数
files = sys.argv[4:]  # 输入的多个csv文件
param_A = sys.argv[1]  # 参数A
param_B = sys.argv[2]  # 参数B
param_C = sys.argv[3]  # 参数C

# 创建大图
fig, axs = plt.subplots(len(files), 1, figsize=(8, 6 * len(files)))

# 循环处理每个CSV文件
for i, file in enumerate(files):
    with open(file, 'r') as f:
        reader = csv.reader(f)
        
        # 获取CSV文件名作为子图标题
        title = file.split(".csv")[0]
        
        # 读取首行作为每列标题
        headers = next(reader)
        
        # 找到参数A和参数B对应的列索引
        idx_A = headers.index(param_A)
        idx_B = headers.index(param_B)
        idx_C = headers.index(param_C)

        # 初始化x和y的列表
        x_vals = []
        y_vals_A = []
        y_vals_B = []
        y_vals_C = []

        # 读取CSV文件中的数据
        for xi, row in enumerate(reader):
            x_vals.append(xi)
            # 从参数A和参数B对应的列中提取第一个浮点数
            y_vals_A.append(float(re.findall(r"\d+\.\d+|\d+", row[idx_A])[0]))
            y_vals_B.append(float(re.findall(r"\d+\.\d+|\d+", row[idx_B])[0]))
            y_vals_C.append(float(re.findall(r"\d+\.\d+|\d+", row[idx_C])[0]))
            

        # 在大图中添加子图
        ax1 = axs[i]

        color = 'tab:red'
        ax1.set_xlabel('timestamp(s)')
        lineA, = ax1.plot(x_vals, y_vals_A, color=color, label=param_A)
        ax1.set_ylabel(param_A, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        lineB, = ax2.plot(x_vals, y_vals_B, color=color, label=param_B)
        ax2.set_ylabel(param_B, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        ax3 = ax1.twinx()
        color = 'tab:green'
        lineC, = ax3.plot(x_vals, y_vals_C, color=color, label=param_C)
        #ax2.set_ylabel(param_C, color=color)
        #ax2.tick_params(axis='y', labelcolor=color)
        ax3.spines['right'].set_position(('outward', 60))  # 调整第三个y轴的位置
        ax3.set_ylabel(param_C)
        #ax3.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        #min_y = min(y_vals_C)
        #max_y = max(y_vals_C)
        #ax3.axhline(min_y, color='green', linestyle='--')
        #ax3.axhline(max_y, color='green', linestyle='--')

        ax1.set_title(title)
        #ax1.xaxis.set_major_locator(plt.MultipleLocator(1))

# 添加鼠标移动事件处理程序
def on_move(event):
    for ax in axs.flat:
        if event.inaxes == ax:
            x_pos = event.xdata

            if x_pos is not None:
                # 清除之前的注释
                ax.texts.clear()

                # 获取鼠标位置对应的索引
                index = int(np.round(x_pos * (len(x) - 1) / ax.get_xlim()[1]))

                # 获取x和所有y的值
                x_val = x[index]
                y_vals = [ax.lines[i].get_ydata()[index] for i in range(len(ax.lines))]

                # 构建注释字符串
                x_str = f'x={x_val:.2f}'
                y_strs = [f'y{i+1}={y:.2f}' for i, y in enumerate(y_vals)]
                annotation_str = '\n'.join([x_str] + y_strs)

                # 添加新的注释
                ax.annotate(annotation_str, (x_val, y_vals[0]), xytext=(5, 5), textcoords='offset points', ha='left', va='bottom')

    # 更新图形
    fig.canvas.draw()

fig.canvas.mpl_connect('motion_notify_event', on_move)

# 调整子图布局
plt.tight_layout()

# 保存图片
plt.savefig('combined_plot.png')

# 显示图片
plt.show()
