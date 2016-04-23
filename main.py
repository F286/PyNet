import numpy as np
import matplotlib.pyplot as plt
import random

class Neuron:
    value = 0.0
    gradient = 0.0

    weights = np.array([])

    def __init__(self, shapePrev):
        self.value = (random.random() - 0.5) * 2
        self.weights = np.random.randn(shapePrev, shapePrev)

    def forward(self, layerPrev):
        valuesPrev = [[n.value for n in row] for row in layerPrev]
        self.value = np.sum(np.dot(valuesPrev, self.weights))
        self.value = max(0, self.value)

numLayers = 2
shape = 4;

layer1 = [[Neuron(0) for x in range(shape)] for y in range(shape)]
layer2 = [[Neuron(shape) for x in range(shape)] for y in range(shape)]

[[n.forward(layer1) for n in row] for row in layer2]

imgData = [(.2, 1.0, 1.0)
           for x in range(numLayers)
           for y in range(shape)]

imgNp = np.array(imgData)
fig, axes = plt.subplots(shape, shape, figsize=(12, 6),
                         subplot_kw={'xticks': [], 'yticks': []})

fig.subplots_adjust(hspace=0.3, wspace=0.05)

for x, y in np.ndenumerate(layer1):
    w = layer1[x][y].weights
    axes.flat[x * shape + y].imshow(w, interpolation='nearest')



# index = 0;
# for axis, layer in zip(axes.flat, [layer1]):#, layer2]):
#     index += 1
#     v = []
#     for row in layer:
#         v.append([n.value for n in row])
#     axis.imshow(np.array(v), interpolation='nearest')
#     axis.set_title(index)

plt.show()