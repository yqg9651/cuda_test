#!/bin/bash

rm $1/sta*

python3 test $1/after.d $1/before.d $1/sta.d
python3 test $1/after.r $1/before.r $1/sta.r
