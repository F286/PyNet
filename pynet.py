import numpy as np

val1 = np.pad(np.random.randn(2, 2), 1, 'constant')
val2 = np.pad(np.random.randn(2, 2), 1, 'constant')
weight2 = np.random.randn(val2.shape[0], val2.shape[1], 3, 3)
d_weight2 = np.zeros(shape=(val2.shape[0], val2.shape[1], 3, 3))

# print val1
# print weight2[0, 0]
# print(val1)
# print(val2)

for p, v in np.ndenumerate(val2[1:-1, 1:-1]):
    # print p[0] + 1, p[1] + 1
    val2[p[0] + 1, p[1] + 1] = np.sum(val1[ p[0]:p[0]+3, p[1]:p[1]+3 ] * weight2[p])
    # val2[np.add(p, 1)] = 2

# val2[1, 1] = 2

print(val1)
print(val2)

# for p, v in np.ndenumerate(val1):
#     min = np.fmax(0, np.add(p, -1))
#     max = np.add(p, 2)
#     valPrev = val1[min[0]:max[0], min[1]:max[1]]
#     val2 = np.sum(np.multiply(weight2[p], valPrev))
#
#
# print(weight2)