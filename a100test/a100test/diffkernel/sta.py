#!/usr/bin/python
import re
import numpy as np
import os
import sys

if len(sys.argv) <= 2:
  print("please enter file path!")
  exit(0)

fn = sys.argv[1]
func = sys.argv[2]

with open(fn) as fin:
  lines = fin.readlines()

result = []
for line in lines:
  line = line[:-1]
  if func in line:
    r = re.findall(r"\d+\.?\d*", line)
    result = np.append(result, float(r[len(r)-1]))

print("mean: %f us" %np.mean(result))
print("std: %f us" %np.std(result))
print("min: %f us" %np.min(result))
print("max: %f us" %np.max(result))
