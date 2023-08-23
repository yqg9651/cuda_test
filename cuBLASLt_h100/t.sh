#!/bin/bash

my_array=(
	"./build/LtHSHgemmStridedBatchSimple/sample_cublasLt_LtHSHgemmStridedBatchSimple -P 0 -T, fp16_half_allzero"
	#"./build/LtHSHgemmStridedBatchSimple/sample_cublasLt_LtHSHgemmStridedBatchSimple -P 2 -T, fp16_half_random"
	"./build/LtHSHgemmStridedBatchSimple/sample_cublasLt_LtHSHgemmStridedBatchSimple -P 1 -T, fp16_half_maxpower"
)

for item in "${my_array[@]}"; do
  command=$(echo "$item" | cut -d',' -f1)
  output_file=$(echo "$item" | cut -d',' -f2)
  echo "$command"
  rm $output_file.csv $output_file.log $output_file.trace;

  # power data
  ./gpu_monitor.sh "$output_file" &
  $command $((1 << 16)) 2>&1 | tee $output_file.log
  sleep 8
  pkill -f nvidia-smi
  
  # trace data
  #sleep 10
  #for((i=0;i<8192;i+=8))
  #do
  #	$command $i | tee -a $output_file.trace
  #	sleep 5
  #done
done
