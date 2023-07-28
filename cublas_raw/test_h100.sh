#!/bin/bash

# 定义数组
my_array=(
	"./cublasBenchMatmul -P=hsh  -ta=1 -tb=0 -A=1 -B=0 -p=1 -m=8192 -n=8192 -k=8192 -T=10000, h100_pcie_random_fp16_float"
	"./cublasBenchMatmul -P=bfsbf  -ta=1 -tb=0 -A=1 -B=0 -p=1 -m=8192 -n=8192 -k=8192 -T=10000, h100_pcie_random_bf16"
	"./cublasBenchMatmul -P=sss_fast_tf32  -ta=1 -tb=0 -A=1 -B=0 -p=1 -m=8192 -n=8192 -k=8192 -T=10000, h100_pcie_random_tf32"
	"./cublasBenchMatmul -P=hsh  -ta=1 -tb=0 -A=1 -B=0 -p=0 -m=8192 -n=8192 -k=8192 -T=10000, h100_pcie_allzero_fp16_float"
	"./cublasBenchMatmul -P=bfsbf  -ta=1 -tb=0 -A=1 -B=0 -p=0 -m=8192 -n=8192 -k=8192 -T=10000, h100_pcie_allzero_bf16"
	"./cublasBenchMatmul -P=sss_fast_tf32  -ta=1 -tb=0 -A=1 -B=0 -p=0 -m=8192 -n=8192 -k=8192 -T=10000, h100_pcie_allzero_tf32"
	"./cublasBenchMatmul -P=hsh  -ta=1 -tb=0 -A=1 -B=0 -p=1 -m=8192 -n=8192 -k=8192 -T=10000 -maxPower=1, h100_pcie_random_fp16_float_max"
	"./cublasBenchMatmul -P=bfsbf  -ta=1 -tb=0 -A=1 -B=0 -p=1 -m=8192 -n=8192 -k=8192 -T=10000 -maxPower=1, h100_pcie_random_bf16_max"
	"./cublasBenchMatmul -P=sss_fast_tf32  -ta=1 -tb=0 -A=1 -B=0 -p=1 -m=8192 -n=8192 -k=8192 -T=10000 -maxPower=1, h100_pcie_random_tf32_max"
)

# 循环遍历数组
for item in "${my_array[@]}"; do
  # 提取命令和输出文件名
  command=$(echo "$item" | cut -d',' -f1)
  output_file=$(echo "$item" | cut -d',' -f2)
  # 执行命令并将stdout输出到文件
  echo "$command"
  ./gpu_monitor.sh "$output_file" &
  $command | tee $output_file.log
  kill %1
done
