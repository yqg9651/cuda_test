#!/bin/bash

data_type=$1
#python3 multi_draw.py " power.draw.instant [W]" " clocks.current.sm [MHz]" " utilization.gpu [%]" \
#	${data_type}_maxpower.csv \
#	${data_type}_allzero.csv
python3 multi_draw.py " power.draw.instant [W]" " clocks.current.sm [MHz]" " temperature.gpu" ${data_type}.csv
	#${data_type}_random.csv
