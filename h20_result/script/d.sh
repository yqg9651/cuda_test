#!/bin/bash

data_type=$1
#python3 multi_draw.py " power.draw.instant [W]" " clocks.current.sm [MHz]" " utilization.gpu [%]" ${data_type}.csv
#python3 multi_draw.py " power.draw.instant [W]" " clocks.current.sm [MHz]" " temperature.gpu" ${data_type}.csv
python3 multi_draw.py " power.draw [W]" " clocks.current.sm [MHz]" ${data_type}.csv
