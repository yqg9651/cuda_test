#!/bin/bash

#python3 multi_draw.py " power.draw.instant [W]" " clocks.current.sm [MHz]" random_gemm_fp16_half.csv allzero_gemm_fp16_half.csv
python3 multi_draw.py " power.draw.instant [W]" " clocks.current.sm [MHz]" " temperature.gpu" random_gemm_bf16.csv allzero_gemm_bf16.csv
