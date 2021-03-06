import numpy as np
import matplotlib.pyplot as plt
import math as math

class ConvLayer:
    width = 8
    height = 8
    value = [0.0]
    gradient = [0.0]
    weight = [0.0]
    # 3 = 3x3, 5 = 5x5
    kernel = 3

    def __init__(self):
        self.value = np.random.randn(self.width * self.height)
        self.gradient = np.random.randn(self.width * self.height * self.kernel ** 2)
        self.weight = np.random.randn(self.width * self.height * self.kernel ** 2)

        l = self
        for x, y in [(x, y) for y in range(l.height) for x in range(l.width)]:
            l.value[x + y * l.width] = 2 - math.sqrt((x - 4) ** 2 + (y - 4) ** 2) ;

    def clear(self):
        for i in range(len(self.gradient)):
            self.gradient[i] = 0

    # writes values to self, from prev
    def forward(self, input):
        l = self
        for x, y in [(x,y) for y in range(l.height) for x in range(l.width)]:
            l.value[x + y * l.width] = 0
            neuronOffset = (x + y * l.width) * l.kernel ** 2
            for _x, _y in [(_x, _y) for _y in range(l.kernel) for _x in range(l.kernel)]:
                neuronValue = x + y * l.width
                neuronSub = neuronOffset + _x + _y * l.kernel
                inputValue = (x + _x - (l.kernel-1)/2) + (y + _y - (l.kernel-1)/2) * l.width

                if 0 < inputValue < len(input.value):
                    l.value[neuronValue] += l.weight[neuronSub] * input.value[inputValue]

    # writes to the derivatives of prev, from self
    def backward(self, input):
        l = self
        for x, y in [(x, y) for y in range(input.height) for x in range(input.width)]:
            neuronOffset = (x + y * l.width) * l.kernel ** 2
            for _x, _y in [(_x, _y) for _y in range(l.kernel) for _x in range(l.kernel)]:
                neuronValue = x + y * l.width
                neuronSub = neuronOffset + _x + _y * l.kernel
                inputValue = (x + _x - (l.kernel-1)/2) + (y + _y - (l.kernel-1)/2) * l.width

                if 0 < inputValue < len(input.value):
                    input.gradient[neuronSub] += input.value[inputValue] * l.weight[neuronSub]

        # l = self
        # for x, y in [(x,y) for y in range(l.height) for x in range(l.width)]:
        #     l.value[x + y * l.width] = 0
        #     root = (x + y * l.width) * l.kernel ** 2
        #     for _x, _y in [(_x, _y) for _y in range(l.kernel) for _x in range(l.kernel)]:
        #         i = root + _x + _y * l.kernel
        #         iPrev = (x + _x - (l.kernel-1)/2) + (y + _y - (l.kernel-1)/2) * l.width
        #
        #         if 0 < iPrev < len(prev.value):
        #             # l.value[x + y * l.width] += l.weight[i] * prev.value[iPrev]
        #             prev.gradient[i] += l.value[x + y * l.width]
        #             prev.gradient[i] = -1


np.random.seed(0)

layer = []
layer.append(ConvLayer())
layer.append(ConvLayer())
layer.append(ConvLayer())

# layer[0].weight =

for i in layer:
    i.clear()

last = layer[len(layer) - 1]
for i in range(len(last.gradient)):
    if i % (last.height * 9) >30:
        last.gradient[i] = 1
    else:
        last.gradient[i] = 0

for i in range(len(layer) - 1):
    layer[i + 1].forward(layer[i])

for i in range(len(layer) - 1):
    layer[i + 1].backward(layer[i])



# [layer[i + 1].forward(layer[i]) for i in layer[1:]]
# for i, l in enumerate(layer, start=1):
#     layer[i].forward(layer[i-1])

# layer[1].forward(layer[0])
# layer[2].forward(layer[1])

# val1 = np.random.randn(32, 32)
# val2 = np.random.randn(32, 32)
# weight2 = np.random.randn(val2.shape[0], val2.shape[1], 3, 3)
#
# val3 = np.random.randn(32, 32)
# mean3 = 0.0
# std3 = 0.0
#
# val4 = np.random.randn(32, 32)
# softmaxDivide4 = 0.0
#
# # layer 2 convolution
# val1Pad = np.pad(val1, 1, 'constant')
# for p, v in np.ndenumerate(val2[1:-1, 1:-1]):
#     val2[p[0] + 1, p[1] + 1] = np.sum(val1Pad[ p[0]:p[0]+3, p[1]:p[1]+3 ] * weight2[p])
#
# # layer 3 normalize
# mean3 = np.average(val2)
# std3 = np.sqrt(np.sum((val2 - mean3) ** 2))
#
# val3 = (val2 - mean3) / std3
#
# # layer 4 softmax
# softmaxDivide4 = np.sum(np.exp(val3))
# val4 = np.exp(val3) / softmaxDivide4
#
#
# # layer 4 gradient
# gradVal4 = np.ones(val4.shape)
# gradVal4 *= np.exp(val4) / softmaxDivide4
#
# # layer 3 gradient
# gradVal3 = gradVal4 / std3
#
# # layer 2 gradient
#
# # val1Pad = np.pad(val1, 1, 'constant')
# # for p, v in np.ndenumerate(val2[1:-1, 1:-1]):
# #     val2[p[0] + 1, p[1] + 1] = np.sum(val1Pad[ p[0]:p[0]+3, p[1]:p[1]+3 ] * weight2[p])
#
# # gradWeight2 = gradVal3
# gradWeight2 = np.random.rand(32, 32) / 3
# gradWeight2 = np.zeros(gradVal3.shape)
# for x in range(weight2.shape[0]):
#     for y in range(weight2.shape[1]):
#         gradWeight2[x, y] =
#         # gradWeight2 = weight2 * gradVal3

# print gradWeight2

# print l[0].value

def displayValues(l):
    r = np.zeros((l.width, l.height, 3))
    for x, y in [(x, y) for y in range(l.height) for x in range(l.width)]:
        # root = (x + y * l.width) * l.kernel ** 2
        i = x + y# + _x + _y * l.kernel
        c = (max(l.value[i], 0), 0, max(-l.value[i], 0))
        r[x, y] = c
    return r

def displayWeights(l):
    r = np.zeros((l.width * l.kernel, l.height * l.kernel, 3))
    for x, y in [(x, y) for y in range(l.height) for x in range(l.width)]:
        root = (x + y * l.width) * l.kernel ** 2
        for _x, _y in [(_x, _y) for _y in range(l.kernel) for _x in range(l.kernel)]:
            i = root + _x + _y * l.kernel
            c = (max(l.weight[i], 0), 0, max(-l.weight[i], 0))
            r[x * l.kernel + _x, y * l.kernel + _y] = c
    return r

def displayGradients(l):
    r = np.zeros((l.width * l.kernel, l.height * l.kernel, 3))
    for x, y in [(x, y) for y in range(l.height) for x in range(l.width)]:
        root = (x + y * l.width) * l.kernel ** 2
        for _x, _y in [(_x, _y) for _y in range(l.kernel) for _x in range(l.kernel)]:
            i = root + _x + _y * l.kernel
            c = (max(-l.gradient[i], 0), max(l.gradient[i], 0), 0)
            r[x * l.kernel + _x, y * l.kernel + _y] = c
    return r

# def colorize(val):
#     ret = np.zeros((val.shape[0], val.shape[1], 3))
#     for x in range(val.shape[0]):
#         for y in range(val.shape[1]):
#             v = val[x, y]
#             ret[x, y] = (max(v, 0), 0, max(-v, 0))
#     return ret
#
# def colorizeWeight(val, grad):
#     ret = np.zeros((val.shape[0] * 4, val.shape[1] * 4, 3))
#     for x in range(val.shape[0]):
#         for y in range(val.shape[1]):
#             for subx in range(3):
#                 for suby in range(3):
#                     v = val[x, y, subx, suby]
#                     g = grad[x, y]
#                     ret[x * 4 + subx, y * 4 + suby] = (max(v, 0), abs(g), max(-v, 0))
#     return ret

# print layer[2].weight

# Plot the grid
fig, ax = plt.subplots(3, len(layer))
for i, lay in enumerate(layer):
    ax[0, i].imshow(displayValues(lay), interpolation='none')
    ax[1, i].imshow(displayWeights(lay), interpolation='none')
    ax[2, i].imshow(displayGradients(lay), interpolation='none')
# ax[0].imshow(colorize(val2), interpolation='none')
# ax[1].imshow(colorizeWeight(weight2, gradWeight2), interpolation='none')
plt.gray()
plt.show()
