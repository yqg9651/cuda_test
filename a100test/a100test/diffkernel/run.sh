#!/bin/bash

export PATH=/usr/local/cuda/bin:$PATH
sudo cpufreq-set -g performance

nvcc test.cu -ptx
nvcc performance.cu -o performance -lcuda

echo "---[pipeline] Kernel"
./performance 0 0 8192
sleep 1
echo ""
echo "---[pipeline] AsyncDMA"
./performance 0 1 8192
sleep 1
echo ""
echo "---[pipeline] HostFunc"
./performance 0 2 8192
sleep 1

echo ""
echo "---[delay] Kernel"
./performance 1 0 8192
sleep 1
echo ""
echo "---[delay] AsyncDMA"
./performance 1 1 8192
sleep 1
echo ""
echo "---[delay] HostFunc"
./performance 1 2 8192
sleep 1

echo ""
echo "---[switch cost] Kernel and DMA"
for ((i=1;i<10;i++))
do
    echo $i time test
    ./performance 2 0 128
done

echo ""
echo "---[graph] struct DKD"
./performance 3 0 16
sleep 1

echo ""
echo "---[graph] struct DKKKKD"
./performance 4 0 16

echo "All Test Finish"



