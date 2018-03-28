import numpy as np


def sigmoid(x, deriv=False):
    if deriv:
        return x * (1 - x)  # This x is actually sigmoid(x), not the ind var x

    return 1 / (1 + np.exp(-x))


def relu(x, deriv=False):
    if deriv:
        outlist = []
        xflat = x.flatten()
        for xp in xflat:
            if xp > 0:
                outlist.append(1)
            else:
                outlist.append(0)
            return outlist
    outlist = []
    xflat = x.flatten()
    for xp in xflat:
        if xp > 0:
            outlist.append(xp)
        else:
            outlist.append(0)
    outlist = np.array(outlist)
    outlist = outlist.reshape(x.shape)
    return outlist


def relumod(x, deriv=False):
    if deriv:
        outlist = []
        xflat = x.flatten()
        for xp in xflat:
            if xp > 0:
                outlist.append(1)
            else:
                outlist.append(0.01)
            return outlist
    outlist = []
    xflat = x.flatten()
    for xp in xflat:
        if xp > 0:
            outlist.append(xp)
        else:
            outlist.append(xp*0.01)
    outlist = np.array(outlist)
    outlist = outlist.reshape(x.shape)
    return outlist


nonlin = sigmoid

X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

np.random.seed(1)

layersize = [3, 4, 1]


# randomly initialize our weights with mean 0
syn = [0] * len(layersize)
for i in range(0, len(layersize)-1):
    syn[i] = 2 * np.random.random((layersize[i], layersize[i+1])) - 1

layers = [0] * len(layersize)
lerrors = [0] * len(layersize)
ldeltas = [0] * len(layersize)

for j in range(60000):

    # Feed forward through layers 0, 1, and 2
    layers[0] = X
    for i in range(1, len(layersize)):
        layers[i] = nonlin(np.dot(layers[i-1], syn[i-1]))

    # how much did we miss the target value?
    lerrors[len(lerrors)] = y - layers[len(layers)]
    for i in range(len(layersize), 1, -1):
        # in what direction is the target value?
        # were we really sure? if so, don't change too much.
        ldeltas[i] = lerrors[i] * nonlin(layers[i], deriv=True)

        # how much did each l1 value contribute to the l2 error (according to the weights)?
        lerrors[i-1] = ldeltas[i].dot(syn[i-1].T)

        # # in what direction is the target l1?
        # # were we really sure? if so, don't change too much.
        # ldeltas[i-1] = lerrors[i-1] * nonlin(layers[i-1], deriv=True)

    if j % 10000 == 0:
        print("The error has been decreased to:" + str(np.mean(np.abs(l2_error))))

    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

print(l2)