from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
##샘플-지수평활 비교
w = pd.read_csv('w5.csv')
wheel = pd.read_csv('C:/Users/Home/Desktop/1. Data (2)/20200520195533/20200520195533.csv')
wheel1=wheel['wheel']
wh = w['wheel']
model = SimpleExpSmoothing(np.asarray(wh))
f = model.fit(smoothing_level=0.2)
wheel2 = f.fittedvalues
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(wheel1, color = 'red')
ax.plot(wheel2, color='blue')
plt.title("Simple Exponential Smoothing")
plt.show()