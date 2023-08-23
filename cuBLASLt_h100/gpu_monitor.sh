#!/bin/bash

nvidia-smi --query-gpu=timestamp,utilization.gpu,temperature.gpu,temperature.memory,power.draw,power.draw.instant,clocks.sm,clocks.mem -lms 1 --format=csv -i 0 -f $1.csv &
#stdbuf -oL nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem -lms 1 --format=csv -i 0 2>&1 > $1.clock.csv &
#stdbuf -oL nvidia-smi --query-gpu=timestamp,power.draw,power.draw.instant -lms 1 --format=csv -i 0 2>&1 > $1.power.csv &
#stdbuf -oL nvidia-smi --query-gpu=timestamp,utilization.gpu,temperature.gpu,temperature.memory -lms 1 --format=csv -i 0 2>&1 > $1.temp.csv &
#nvidia-smi -q -d UTILIZATION,TEMPERATURE,POWER,CLOCK,VOLTAGE -lms 1 -i 0 2>&1 > $1.csv
stdbuf -oL nvidia-smi -q -d VOLTAGE -lms 1 -i 0 -f $1.voltage.info &
#nvidia-smi -q -d CLOCK -lms 1 -i 0 2>&1 > $1_clock.csv &
#nvidia-smi -q -d POWER -lms 1 -i 0 2>&1 > $1_power.csv &
