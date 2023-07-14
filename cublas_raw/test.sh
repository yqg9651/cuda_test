#!/bin/bash

# 定义数组
my_array=(
"./cublasMatmulBench -P=hhh  -ta=1 -tb=0 -A=1 -B=0 -p=1 -m=8192 -n=8192 -k=8192 -T=10000, random_gemm_fp16_half"
"./cublasMatmulBench -P=hsh  -ta=1 -tb=0 -A=1 -B=0 -p=1 -m=8192 -n=8192 -k=8192 -T=10000, random_gemm_fp16_float"
"./cublasMatmulBench -P=tst  -ta=1 -tb=0 -A=1 -B=0 -p=1 -m=8192 -n=8192 -k=8192 -T=10000, random_gemm_bf16"
"./cublasMatmulBench -P=sss_fast_tf32  -ta=1 -tb=0 -A=1 -B=0 -p=1 -m=8192 -n=8192 -k=8192 -T=10000, random_gemm_tf32"
"./cublasMatmulBench -P=qqssq  -ta=1 -tb=0 -A=1 -B=0 -p=1 -m=8192 -n=8192 -k=8192 -T=10000, random_gemm_fp8"
"./cublasMatmulBench -P=sss  -ta=1 -tb=0 -A=1 -B=0 -p=1 -m=8192 -n=8192 -k=8192 -T=1000, random_gemm_fp32"
"./cublasMatmulBench -P=bii_imma  -ta=1 -tb=0 -A=1 -B=0 -p=1 -m=8192 -n=8192 -k=8192 -T=1000, random_gemm_int8"

"./cublasMatmulBench -P=hhh  -ta=1 -tb=0 -A=1 -B=0 -p=0 -m=8192 -n=8192 -k=8192 -T=10000, allzero_gemm_fp16_half"
"./cublasMatmulBench -P=hsh  -ta=1 -tb=0 -A=1 -B=0 -p=0 -m=8192 -n=8192 -k=8192 -T=10000, allzero_gemm_fp16_float"
"./cublasMatmulBench -P=tst  -ta=1 -tb=0 -A=1 -B=0 -p=0 -m=8192 -n=8192 -k=8192 -T=10000, allzero_gemm_bf16"
"./cublasMatmulBench -P=sss_fast_tf32  -ta=1 -tb=0 -A=1 -B=0 -p=0 -m=8192 -n=8192 -k=8192 -T=10000, allzero_gemm_tf32"
"./cublasMatmulBench -P=qqssq  -ta=1 -tb=0 -A=1 -B=0 -p=0 -m=8192 -n=8192 -k=8192 -T=10000, allzero_gemm_fp8"
"./cublasMatmulBench -P=sss  -ta=1 -tb=0 -A=1 -B=0 -p=0 -m=8192 -n=8192 -k=8192 -T=1000, allzero_gemm_fp32"
"./cublasMatmulBench -P=bii_imma  -ta=1 -tb=0 -A=1 -B=0 -p=0 -m=8192 -n=8192 -k=8192 -T=1000, allzero_gemm_int8"
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
