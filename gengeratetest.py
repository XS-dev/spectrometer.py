import matplotlib.pyplot as plt
import numpy as np
import math
x = np.linspace(430,480,50)#设置画图范围

quantity1 = 3
quantity2 = 0

sigma1 = 8  #方差
u1 = 455 #均值设置为1
y1 = (1 / (math.sqrt(2*math.pi) /sigma1)) * np.exp(-(x - u1) ** 2 / (2 * (sigma1 ** 2)))


sigma2 = 6  #方差设置为2
u2 = 460 #均值设置为1
y2 = (1 / (math.sqrt(2*math.pi) /sigma2)) * np.exp(-(x - u2) ** 2 / (2 * (sigma2 ** 2)))

y = quantity1* y1+quantity2*y2

plt.figure()
plt.plot(x,y)
plt.show()

np.save('test0',y)