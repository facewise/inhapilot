import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
##샘플-이동평균값 비교
wheel = pd.read_csv('C:/Users/Home/Desktop/1. Data (2)/20200520195533/20200520195533.csv')
wheel1 = pd.read_csv('w5.csv')
wheel2 = wheel1['wheel'].rolling(window=10).mean()

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(wheel['wheel'],color = 'red')
ax.plot(wheel2,color = 'blue')

plt.title("Moving")
plt.show()
