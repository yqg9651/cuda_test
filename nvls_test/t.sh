#!/bin/bash

#nsys profile --gpu-metrics-set=gh100 --gpu-metrics-device=0 --gpu-metrics-frequency=200000 ./alltoall_perf -g 8 -b $((1<<30)) -c 0 -e $((1<<30)) -z 0 -f 2 -n 1 -w 1 | tee nvls_8c_all2all_nsyslog

ncu --replay-mode app-range --target-processes all --set full, --section Nvlink ./all_reduce_perf -g 8 -b $((2<<30)) -c 0 -e $((2<<30)) -z 0 -f 2 -n 100 -w 50 | tee all_reduce.profiler_data
ncu --replay-mode app-range --target-processes all --set full, --section Nvlink ./alltoall_perf -g 8 -b $((2<<30)) -c 0 -e $((2<<30)) -z 0 -f 2 -n 100 -w 50 | tee alltoall.profiler_data
ncu --replay-mode app-range --target-processes all --set full, --section Nvlink ./broadcast_perf -g 8 -b $((2<<30)) -c 0 -e $((2<<30)) -z 0 -f 2 -n 100 -w 50 | tee broadcast.profiler_data
ncu --replay-mode app-range --target-processes all --set full, --section Nvlink ./reduce_perf -g 8 -b $((2<<30)) -c 0 -e $((2<<30)) -z 0 -f 2 -n 100 -w 50 | tee reduce.profiler_data
ncu --replay-mode app-range --target-processes all --set full, --section Nvlink ./all_gather_perf -g 8 -b $((2<<30)) -c 0 -e $((2<<30)) -z 0 -f 2 -n 100 -w 50 | tee all_gather.profiler_data
ncu --replay-mode app-range --target-processes all --set full, --section Nvlink ./hypercube_perf -g 8 -b $((2<<30)) -c 0 -e $((2<<30)) -z 0 -f 2 -n 100 -w 50 | tee hypercube.profiler_data
ncu --replay-mode app-range --target-processes all --set full, --section Nvlink ./reduce_scatter_perf -g 8 -b $((2<<30)) -c 0 -e $((2<<30)) -z 0 -f 2 -n 100 -w 50 | tee reduce_scatter.profiler_data
ncu --replay-mode app-range --target-processes all --set full, --section Nvlink ./scatter_perf -g 8 -b $((2<<30)) -c 0 -e $((2<<30)) -z 0 -f 2 -n 100 -w 50 | tee scatter.profiler_data
ncu --replay-mode app-range --target-processes all --set full, --section Nvlink ./sendrecv_perf -g 8 -b $((2<<30)) -c 0 -e $((2<<30)) -z 0 -f 2 -n 100 -w 50 | tee sendrecv.profiler_data
#./sendrecv_perf -g 2 -b $((1<<30)) -c 0 -e $((1<<30)) -z 0 -f 2 -n 1 -w 1 | tee nvls_2c_sendrecv_nsyslog
#ncu --target-processes all --set full, --section Nvlink ./broadcast_perf -g 8 -b $((1<<29)) -c 0 -e $((1<<29)) -z 0 -f 2 -n 1 -w 1 | tee nvls_8c_all2all_nculog
