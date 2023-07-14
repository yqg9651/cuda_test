import pandas as pd
import matplotlib.pyplot as plt
import sys

df = pd.read_csv(sys.argv[1], skiprows=1, sep=',')

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(x = df['         offset'], y = df['     latency(cycles)'])
plt.xlabel("offset")
plt.ylabel("latency(cycles)")

plt.show()
