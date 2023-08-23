#!/bin/bash

data_type=fp16_half
python3 multi_draw.py " power.draw.instant [W]" " clocks.current.sm [MHz]" " utilization.gpu [%]" ${data_type}_random.csv ${data_type}_allzero.csv
