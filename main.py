import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image

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
        # wires = np.dot([i.value for i in layerPrev], self.weights)
        # self.value = max(0, np.sum(wires))

numLayers = 2
# numNeurons = 16
shape = 4;

layer1 = [[Neuron(0) for x in range(shape)] for y in range(shape)]
layer2 = [[Neuron(shape) for x in range(shape)] for y in range(shape)]
# layer1 = [Neuron(0) for i in range(numNeurons)]
# layer2 = [Neuron(numNeurons) for i in range(numNeurons)]

[[n.forward(layer1) for n in row] for row in layer2]
# [layer2. for x, y in ndenumerate(layer2)]
# [n.forward(layer1) for n in layer2]

# imgData = []w
# for x in range(10):
#     for y in range(100):
#         imgData.append((1, 1, 1))

imgData = [(.2, 1.0, 1.0)
           for x in range(numLayers)
           for y in range(shape)]
# imgData = [(layer1[y].value, 1.0, 1.0)
#            for x in range(numLayers)
#            for y in range(shape)]

# print(imgData[0])

# img = Image.new('HSV', (10, 100),)
# img.putdata(imgData)

imgNp = np.array(imgData)
# img.save('image.png')
# img.show()


# layer2.forward(layer1)


# print(layer1.__len__())

# each neuron
# value = np.random.random((numLayers, numNeurons))
# # layer number, each neuron
# weight = np.random.random((numNeurons - 1, numNeurons))
#
# forward(value, weight, 1)

# x = [i for i, n in enumerate(layer2)]
# y = [n.value for n in layer2]
#
# y.sort()

# x = np.arange(0, 4 * np.pi, 0.1)
# y = np.sin(x)
# z = np.tanh(x)
#
# nums = [0,1,2,3,4,5]
# squares = [x ** 2 for x in nums]

# Plot the points using matplotlib
# plt.plot(x, y)
# circle1=plt.Circle((0,0),.2,color='r')
# # plt.plot(x, z)
# # plt.plot(nums, squares)
# # plt.plot(value, weight)
# # plt.plot()
# # plt.hist([n.value for n in layer2], 20, (-5, 5))
# plt.cir
# plt.imshow(imgNp, aspect='auto', interpolation='none')
# plt.show()  # You must call plt.show() to make graphics appear.


# methods = [None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16',
#            'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
#            'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']

#grid = np.random.rand(4, 4)

fig, axes = plt.subplots(1, 2, figsize=(12, 6),
                         subplot_kw={'xticks': [], 'yticks': []})
#print(axes.flat)
fig.subplots_adjust(hspace=0.3, wspace=0.05)

# print(layer1[0])

for i, ax in enumerate(axes.flat):
    v = []
    for row in layer1:
        v.append([n.value for n in row])
    ax.imshow(np.array(v), interpolation='nearest')

# for ax, interp_method in zip(axes.flat, methods):
#     ax.imshow(grid, interpolation=interp_method)
#     ax.set_title(interp_method)

plt.show()