import re
import matplotlib.pyplot as plt
import sys

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
            
            # 查找包含 "mV" 的行
            if 'mV' in line:
                # 提取前一个浮点数作为 y 坐标值
                y_value = extract_float_value(line.split('mV')[0])
                if y_value is not None:
                    x_values.append(len(x_values) * 4)  # x 坐标值递增4
                    y_values.append(y_value)

    # 绘制图形
    plt.plot(x_values, y_values, marker='+')
    plt.xlabel('timestamp(ms)')
    plt.ylabel('voltage(mV)')
    plt.title(file_path.split(".")[0])
    plt.show()

if __name__ == '__main__':
    main()
