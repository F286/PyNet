import numpy as np

val1 = np.random.randn(5, 5)

for p, v in np.ndenumerate(val1):
    min = np.fmax(0, np.add(p, -1))
    max = np.add(p, 2)
    print np.sum(val1[min[0]:max[0], min[1]:max[1]])