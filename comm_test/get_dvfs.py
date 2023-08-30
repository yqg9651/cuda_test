import csv
import re
import subprocess

# 执行命令并返回输出结果
def execute_command(command):
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        return result.stdout.strip()
    else:
        return None

# 获取最大频率
def get_max_freq():
    command = "nvidia-smi --query-gpu=clocks.max.sm --format=csv -i 0"
    output = execute_command(command)
    if output:
        match = re.search(r'(\d+)', output, re.MULTILINE)
        if match:
            return int(match.group(1))
    
    return None

def get_mem_freq():
    command = "nvidia-smi --query-gpu=clocks.mem --format=csv -i 0"
    output = execute_command(command)
    if output:
        match = re.search(r'(\d+)', output, re.MULTILINE)
        if match:
            return int(match.group(1))
    
    return None

def get_video_freq():
    command = "nvidia-smi --query-gpu=clocks.video --format=csv -i 0"
    output = execute_command(command)
    if output:
        match = re.search(r'(\d+)', output, re.MULTILINE)
        if match:
            return int(match.group(1))
    
    return None

def get_graphic_freq():
    command = "nvidia-smi --query-gpu=clocks.gr --format=csv -i 0"
    output = execute_command(command)
    if output:
        match = re.search(r'(\d+)', output, re.MULTILINE)
        if match:
            return int(match.group(1))
    
    return None

# 获取电压
def get_voltage(freq):
    command = f"nvidia-smi -d VOLTAGE -q -i 0"
    output = execute_command(command)
    if output:
        match = re.search(r'(\d+\.\d+)\s*mV', output, re.MULTILINE)
        if match:
            voltage = float(match.group(1))
            return voltage
    
    return None

# 写入CSV文件
def write_to_csv(data):
    with open('gpu_dvfs.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['SM Frequency (MHz)', 'Voltage (mV)', 'Mem Frequency (MHz)', 'Video Freqeuncy (MHz)', 'Graphic Frequency (MHz)'])
        writer.writerows(data)

def main():
    max_freq = get_max_freq()
    print("Max Frequency:", max_freq)

    if max_freq:

        freq = max_freq
        data = []

        while freq >= 0:
            command = f"sudo nvidia-smi -lgc {freq},{freq} -m 1 -i 0"
            execute_command(command)
            
            command = "nvidia-smi --query-gpu=clocks.sm --format=csv -i 0"
            output = execute_command(command)
            if output:
                match = re.search(r'(\d+)', output, re.MULTILINE)
                if match and int(match.group(1)) == freq:
                    voltage = get_voltage(freq)
                    mem_freq = get_mem_freq()
                    video_freq = get_video_freq()
                    graphic_freq = get_graphic_freq()
                    if voltage:
                        data.append([freq, voltage, mem_freq, video_freq, graphic_freq])
                        freq -= 5
                        continue

            freq -= 5

        if data:
            write_to_csv(data)

        command = "sudo nvidia-smi -rgc"
        execute_command(command)

if __name__ == '__main__':
    main()

