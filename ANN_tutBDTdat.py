import numpy as np


def sigmoid(x, deriv=False):
    if deriv:
        return x * (1 - x)  # This x is actually sigmoid(x), not the ind var x

    return 1 / (1 + np.exp(-x))


def relu(x, deriv=False):
    if type(x) == list:
        x = np.array(x)
    if deriv:
        outlist = []
        xflat = x.flatten()
        for xp in xflat:
            if xp > 0:
                outlist.append(1)
            else:
                outlist.append(0)
        outlist = np.array(outlist)
        outlist = outlist.reshape(x.shape)
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
    slope = 0.1
    if type(x) == list:
        x = np.array(x)
    if deriv:
        outlist = []
        xflat = x.flatten()
        for xp in xflat:
            if xp > 0:
                outlist.append(1)
            else:
                outlist.append(slope)
        outlist = np.array(outlist)
        outlist = outlist.reshape(x.shape)
        return outlist
    outlist = []
    xflat = x.flatten()
    for xp in xflat:
        if xp > 0:
            outlist.append(xp)
        else:
            outlist.append(xp * slope)
    outlist = np.array(outlist)
    outlist = outlist.reshape(x.shape)
    return outlist


def add_bias(x, how='column'):
    if how == 'column':
        X_new = np.ones((x.shape[0], x.shape[1] + 1))
        X_new[:, 1:] = x
    elif how == 'row':
        X_new = np.ones((x.shape[0] + 1, x.shape[1]))
        X_new[1:, :] = x
    return X_new


###################################### Import data ##############################################

useadvBDTdat = False
usesimBDTdat = True

if useadvBDTdat:

    data = np.genfromtxt('/Users/idastoustrup/Documents/Dropbox/Skole/Fysiske Fag/4. year/AnvStat2/Exercises/Week 6/'
                         'BDTs/BDT_16var.txt')
    data = data.T
    data = data[1:len(data)]
    data = data.T
    endtrain = int(len(data)/2)
    datatest = data[endtrain:len(data)]
    datatrain = data[0:endtrain]
    Ndhalf = len(datatrain)/2
    y = [0, 1]*int(Ndhalf)
    y = np.array(y)
    y = y.reshape(len(datatrain), 1)
    X = np.array(datatrain)

elif usesimBDTdat:

    datateb = np.genfromtxt('/Users/idastoustrup/Documents/Dropbox/Skole/Fysiske Fag/4. year/AnvStat2/Exercises/Week 6/'
                            'BDTs/BDT_background_test.txt')
    datatrb = np.genfromtxt('/Users/idastoustrup/Documents/Dropbox/Skole/Fysiske Fag/4. year/AnvStat2/Exercises/Week 6/'
                            'BDTs/BDT_background_train.txt')
    datates = np.genfromtxt('/Users/idastoustrup/Documents/Dropbox/Skole/Fysiske Fag/4. year/AnvStat2/Exercises/Week 6/'
                            'BDTs/BDT_signal_test.txt')
    datatrs = np.genfromtxt('/Users/idastoustrup/Documents/Dropbox/Skole/Fysiske Fag/4. year/AnvStat2/Exercises/Week 6/'
                            'BDTs/BDT_signal_train.txt')

    X = np.concatenate((datatrb, datatrs), axis=0)
    y = np.append(np.zeros(len(datatrb)), np.ones(len(datatrs)))
    y = y.reshape(len(X), 1)

else:

    X = np.array([[0, 0, 0, 1],
                 [0, 1, 0, 0],
                 [1, 0, 0, 0],
                 [0, 0, 1, 0],
                 [0, 1, 1, 0],
                 [1, 1, 1, 0],
                 [0, 0, 1, 1]])

    y = np.array([[1],
                  [1],
                  [1],
                  [1],
                  [0],
                  [0],
                  [0]])

# Data normalisation

for i in range(len(X[0])):
    minx = min(X.T[i])
    maxx = max(X.T[i])
    rangex = abs(minx) + maxx
    X.T[i] = X.T[i]/rangex

####################################### Network settings ###############################################

np.random.seed(1)

nonlin = sigmoid
LR = 0.01
layersize = [len(X[0]), len(X[0])*4, len(X[0])*3, 1]

###################################### Train the network #########################################################

# Bias

bias = False

if bias:
    X = add_bias(X)

# randomly initialize our weights with mean 0

if bias:

    weights = [0] * len(layersize)
    for i in range(0, len(layersize)-1):
        weights[i] = 2 * np.random.random((layersize[i] + 1, layersize[i + 1])) - 1
else:
    weights = [0] * len(layersize)
    for i in range(0, len(layersize) - 1):
        weights[i] = 2 * np.random.random((layersize[i], layersize[i + 1])) - 1

NLayers = len(layersize)

layers = [0] * NLayers
lerrors = [0] * NLayers
ldeltas = [0] * NLayers

Nit = 15000

for j in range(Nit):

    ############ FEED FORWARD ###########

    layers[0] = X
    for i in range(1, NLayers):
        layers[i] = nonlin(np.dot(layers[i-1], weights[i-1]))
        if bias and not i == NLayers - 1:
            layers[i] = add_bias(layers[i])

    ########### FEED BACKWARDS #########

    lerrors[-1] = y - layers[-1]
    for i in range(NLayers - 1, 0, -1):

        ldeltas[i] = lerrors[i] * nonlin(layers[i], deriv=True)

        if bias and i < NLayers - 1:
            ldeltasn = ldeltas[i][:, 1:]
            lerrors[i - 1] = ldeltasn.dot(weights[i - 1].T)
            weights[i - 1] += layers[i - 1].T.dot(ldeltasn) * LR

        else:
            lerrors[i - 1] = ldeltas[i].dot(weights[i - 1].T)
            weights[i - 1] += layers[i - 1].T.dot(ldeltas[i]) * LR

    if j % 1000 == 0:
        print("\nj is %i out of %i"%(j, Nit))
        print("The error has been decreased to:" + str(np.mean(np.abs(lerrors[-1]))))

###################################### Test the network #########################################################

for i in range(5):
    print("\n%i"%i)
    print("Class should have been %i, the ANN guesses %f"%(y[i], layers[-1][i]))

telayers = [0] * NLayers
teX = np.concatenate((datateb, datates), axis=0)

for i in range(len(teX[0])):
    minx = min(teX.T[i])
    maxx = max(teX.T[i])
    rangex = abs(minx) + maxx
    teX.T[i] = teX.T[i]/rangex

telayers[0] = teX

for i in range(1, NLayers):
    telayers[i] = nonlin(np.dot(telayers[i-1], weights[i-1]))

tey = np.append(np.zeros(len(datateb)), np.ones(len(datates)))
tey = tey.reshape(len(teX), 1)

print("The error for the test set is:" + str(np.mean(np.abs(tey - telayers[-1]))))
