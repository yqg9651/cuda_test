#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
echo "test start"
#./cudaLatRd -F stride_32_80MiB_normal.csv             -L 160 -N 100 -S 32 &&
#./cudaLatRd -F stride_32_80MiB_normal_compression.csv -L 80 -N 1000 -S 32 -C &&
#./cudaLatRd -F stride_32_80MiB_const.csv              -L 80 -N 1000 -S 32 -R &&
#./cudaLatRd -F stride_32_80MiB_const_compression.csv  -L 80 -N 1000 -S 32 -R -C &&
#./cudaLatRd -F tlb_normal_2MiB.csv             -L $((32*1024)) -N 100 -S $((2<<20)) &&
#./cudaLatRd -F tlb_normal_32MiB.csv             -L $((32*1024)) -N 100 -S $((32<<20)) &&
./cudaLatRd -F stride_64_120MiB_comp.csv -L $((120 << 20)) -N 200 -S 64 -C &&
./cudaLatRd -F stride_64_120MiB_normal.csv -L $((120 << 20)) -N 200 -S 64 -C &&
echo "test finish"
