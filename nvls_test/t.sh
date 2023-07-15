#!/bin/bash

nsys profile --gpu-metrics-set=gh100 --gpu-metrics-device=0 --gpu-metrics-frequency=200000 ./all_reduce_perf -g 8 -b $((1<<30)) -c 0 -e $((1<<30)) -z 0 -f 2 -n 1 -w 1 | tee nvls_8c_allreduce_nsyslog
