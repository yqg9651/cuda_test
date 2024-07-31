#!/bin/bash

./d.sh $1 &
python3 draw_voltage.py $1.voltage.info &
python3 draw_trace.py $2 $1.trace &
