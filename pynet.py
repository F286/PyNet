import numpy as np
import matplotlib.pyplot as plt
# import glumpy

np.random.seed(0)

val1 = np.random.randn(32, 32)
val2 = np.random.randn(32, 32)
weight2 = np.random.randn(val2.shape[0], val2.shape[1], 3, 3)

val3 = np.random.randn(32, 32)
mean3 = 0.0
std3 = 0.0

val4 = np.random.randn(32, 32)
softmaxDivide4 = 0.0

# layer 2 convolution
val1Pad = np.pad(val1, 1, 'constant')
for p, v in np.ndenumerate(val2[1:-1, 1:-1]):
    val2[p[0] + 1, p[1] + 1] = np.sum(val1Pad[ p[0]:p[0]+3, p[1]:p[1]+3 ] * weight2[p])

# layer 3 normalize
mean3 = np.average(val2)
std3 = np.sqrt(np.sum((val2 - mean3) ** 2))

val3 = (val2 - mean3) / std3

# layer 4 softmax
softmaxDivide4 = np.sum(np.exp(val3))
val4 = np.exp(val3) / softmaxDivide4


# layer 4 gradient
gradVal4 = np.ones(val4.shape)

# print gradVal4
# print np.exp(val4) / softmaxDivide4

gradVal4 *= np.exp(val4) / softmaxDivide4

# layer 3 gradient
gradVal3 = gradVal4 / std3


# window = glumpy.Window(512, 512)
# im = glumpy.Image(z.astype(np.float32), cmap=glumpy.colormap.Grey)

# @window.event
# def on_draw():
#     im.blit(0, 0, window.width, window.height)
# window.mainloop()

def colorize(val):
    ret = np.zeros((val.shape[0], val.shape[1], 3))
    for x in range(val.shape[0]):
        for y in range(val.shape[1]):
            v = val[x, y]
            ret[x, y] = (max(v, 0), 0, max(-v, 0))
    return ret

def colorizeWeight(val):
    ret = np.zeros((val.shape[0] * 4, val.shape[1] * 4, 3))
    for x in range(val.shape[0]):
        for y in range(val.shape[1]):
            for subx in range(3):
                for suby in range(3):
                    v = val[x, y, subx, suby]
                    ret[x * 4 + subx, y * 4 + suby] = (max(v, 0), 0, max(-v, 0))
    return ret

# display = np.zeros((val2.shape[0], val2.shape[1], 3))
# for x in range(display.shape[0]):
#     for y in range(display.shape[1]):
#         display[x, y] = colorize(val2[x, y])

print(val2)

display = colorize(val2)

# colorizeV = np.vectorize(colorize)
#
# print colorizeV(val2)
# # print [colorize(y) for y in val2]
#
# display = colorizeV(val2) #np.vectorize(weight2.reshape(-1, 3, 3))
# for i, d in np.ndenumerate(display):
#     display[i] = (1, 1, 0)

# print(display)

# Plot the grid
fig, ax = plt.subplots(1, 2)
ax[0].imshow(colorize(val2), interpolation='none')
ax[1].imshow(colorizeWeight(weight2), interpolation='none')
plt.gray()
plt.show()

# print gradVal3

# print val2
# print val3
# print val4
# print np.sum(val4)
# print(gradVal4)

# print gradVal4