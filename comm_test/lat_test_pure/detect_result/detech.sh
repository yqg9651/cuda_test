#!/bin/bash

# 循环从256递增到2GB
start=256
end=$((1024 * 1024 * 1024 * 2))
increment=256

for (( size = start; size <= end; size += increment ))
do
    echo "Executing program with size: $size"
    ./../cudaLatMMRd -F $size.csv -L $((size*2)) -G fine_grained -D 8 -S $size
    echo "-------------------------------------"
done
