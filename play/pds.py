import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
##모델-샘플 휠값 비교
wheel1 = pd.read_csv('C:/Users/Home/Desktop/1. Data (2)/20200520195533/20200520195533.csv')
wheel2 = pd.read_csv('w8.csv')
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(wheel1['wheel'],color = 'red')
ax.plot(wheel2['wheel'],color = 'blue')
plt.title("sample-model")
plt.show()