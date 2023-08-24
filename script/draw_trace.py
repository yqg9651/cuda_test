import re
import matplotlib.pyplot as plt
import sys
import os

def extract_float_value(string):
    # 使用正则表达式提取字符串中的浮点数
    match = re.search(r'([-+]?\d*\.\d+|\d+)', string)
    if match:
        return float(match.group())
    else:
        return None

def main():
    # 输入文件路径
    file_path = sys.argv[1]  # 替换为您的输入文件路径

    # 用于存储坐标点的列表
    x_values = []
    y_values = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            
            # 查找包含 "TFlops" 的行
            if 'TFlops' in line:
                # 提取前一个浮点数作为 y 坐标值
                y_value = extract_float_value(line.split('TFlops')[0])
                if y_value is not None:
                    # 提取 ms 后面的浮点数作为 x 坐标值
                    x_value = extract_float_value(line.split('ms')[1])
                    if x_value is not None:
                        x_values.append(x_value)
                        y_values.append(y_value)

    # 从文件名中获取标题
    title = os.path.basename(file_path).split('.')[0]

    # 绘制折线图
    plt.plot(x_values, y_values, marker='o')
    plt.xlabel('x (ms)')
    plt.ylabel('y (TFlops)')
    plt.title(title)
    plt.show()

if __name__ == '__main__':
    main()
