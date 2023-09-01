#!/bin/bash

my_array=(
	"./build/LtHSHgemmStridedBatchSimple/sample_cublasLt_LtHSHgemmStridedBatchSimple -P 0 -T, fp16_half_allzero"
	"./build/LtHSHgemmStridedBatchSimple/sample_cublasLt_LtHSHgemmStridedBatchSimple -P 2 -T, fp16_half_random"
	"./build/LtHSHgemmStridedBatchSimple/sample_cublasLt_LtHSHgemmStridedBatchSimple -P 1 -T, fp16_half_maxpower"
	"./build/LtFp8Matmul/sample_cublasLt_LtFp8Matmul -m $((8192*2)) -n $((8192*2)) -k $((8192*2)) -P 0 -a 1 -T, fp8_e4m3_bf16_fast_mode_allzero"
	"./build/LtFp8Matmul/sample_cublasLt_LtFp8Matmul -m $((8192*2)) -n $((8192*2)) -k $((8192*2)) -P 1 -a 1 -T, fp8_e4m3_bf16_fast_mode_maxpower"
	"./build/LtFp8Matmul/sample_cublasLt_LtFp8Matmul -m $((8192*2)) -n $((8192*2)) -k $((8192*2)) -P 2 -a 1 -T, fp8_e4m3_bf16_fast_mode_random"
	"./build/LtFp8Matmul/sample_cublasLt_LtFp8Matmul -m $((8192*2)) -n $((8192*2)) -k $((8192*2)) -P 0 -a 0 -T, fp8_e4m3_bf16_normal_allzero"
	"./build/LtFp8Matmul/sample_cublasLt_LtFp8Matmul -m $((8192*2)) -n $((8192*2)) -k $((8192*2)) -P 1 -a 0 -T, fp8_e4m3_bf16_normal_maxpower"
	"./build/LtFp8Matmul/sample_cublasLt_LtFp8Matmul -m $((8192*2)) -n $((8192*2)) -k $((8192*2)) -P 2 -a 0 -T, fp8_e4m3_bf16_normal_random"
)

for item in "${my_array[@]}"; do
  command=$(echo "$item" | cut -d',' -f1)
  output_file=$(echo "$item" | cut -d',' -f2)
  echo "$command"
  echo "$output_file"
  rm $output_file.csv $output_file.log $output_file.trace;

  # power data
  ./gpu_monitor.sh "$output_file" &
  for((i=0;i<50;i++)) do $command $((1 << 5)) 2>&1; sleep 1; done | tee $output_file.log
  pkill -f nvidia-smi
  
  # trace data
  #for((i=1;i<2048;i*=2))
  #do
  	#$command $i | tee -a $output_file.trace
  	#sleep 5
  #done
done

./gpu_monitor.sh util_dvfs &
for((i=1024;i<8192*2;i+=1024)) do ./build/LtFp8Matmul/sample_cublasLt_LtFp8Matmul -m $((8192*2)) -n $((8192*2)) -k $i -P 1 -a 1 -T $((1<<8)) 2>&1 ; sleep 1; done | tee util_dvfs.trace
pkill -f nvidia-smi

