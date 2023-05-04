import numpy as np

def d(X,Y):
    return np.sum((X-Y)**2)/(np.sum(X**2)*np.sum(Y**2))

x = np.array([[[1,2,3,4,5],
               [5,6,7,8,5],
               [5,6,7,8,5]],
              [[11,22,23,24,5],
               [25,26,27,28,5],
               [5,6,7,8,5]]])

y = np.array([[[31,32,33,34,5],
               [35,36,37,38,5],
               [5,6,7,8,5]],
              [[41,42,43,44,5],
               [45,46,47,48,5],
               [5,6,7,8,5]]])

xx = x.reshape(2, -1)
yy = y.reshape(2, -1)
dist = np.hypot(*(xx - yy))

print(dist)
print(d(x,y))