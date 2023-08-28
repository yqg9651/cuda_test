import re
import matplotlib.pyplot as plt
import sys
import os

def parse_log_file(log_file, max_tflops):
    x_coords = []
    y_coords = []

    # 正则表达式模式匹配
    #pattern = r'(\d+\.\d+)\s*TFlops.*?ms\s+(\d+\.\d+)\s*$'
    pattern1 = r'(\d+\.\d+)\s+TFlops'
    pattern2 = r'ms=(\d+\.\d+)'

    with open(log_file, 'r') as file:
        for line in file:
            match1 = re.search(pattern1, line)
            match2 = re.search(pattern2, line)
            if match1 and match2:
                y_coord = float(match1.group(1))
                x_coord = float(match2.group(1))
                x_coords.append(x_coord)
                y_coords.append(y_coord)

    # 计算y坐标轴的百分比
    y_coords_percent = [(y / float(max_tflops)) * 100 for y in y_coords]

    # 绘制图形
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    x_sorted, y1_sorted = zip(*sorted(zip(x_coords, y_coords)))
    ax1.plot(x_sorted, y1_sorted, color=color)
    ax1.set_xlabel('timestamp(ms)')
    ax1.set_ylabel('TLOPS', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    x_sorted, y2_sorted = zip(*sorted(zip(x_coords, y_coords_percent)))
    ax2.plot(x_sorted, y2_sorted, color=color)
    ax2.set_ylabel('ratio', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # 绘制y坐标值的最大值虚线
    #ax2.axhline(y=max(y_coords_percent), linestyle='dashed', color='red')
    #plt.gca().axhline(max(y_coords_percent), linestyle='dashed', color='red')
    max_index = y_coords_percent.index(max(y_coords_percent))
    max_x = x_coords[max_index]
    max_y_coords_percent = y_coords_percent[max_index]
    max_y_coords = y_coords[max_index]

    ax2.scatter(max_x, max_y_coords_percent, color='blue', marker='+')
    plt.text(max_x, max_y_coords_percent, f'({max_x}ms, {max_y_coords}TFLOPS, {max_y_coords_percent:.2f}%)', ha='center', va='bottom')

    # draw 90% max performance(TFLOPS)
    p90_max = 0.9 * float(max_tflops)
    if max_y_coords >= p90_max:
      ax1.axhline(y=p90_max, linestyle='dashed', color='red')

    plt.title(os.path.basename(log_file).split(".")[0])
    plt.show()

# 示例用法
log_file = sys.argv[2]  # 替换为你的log文件路径
max_tflops = sys.argv[1]  # 替换为你的最大TFlops值

parse_log_file(log_file, max_tflops)
