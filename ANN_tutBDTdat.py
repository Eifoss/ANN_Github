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


def add_bias(x):
    X_new = np.ones((x.shape[0], x.shape[1] + 1))
    X_new[:, 1:] = x
    return X_new


def diff_cost(yl, lastlyr):
    return yl - lastlyr


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

    testX = np.concatenate((datateb, datates), axis=0)

    for i in range(len(testX[0])):
        minx = min(testX.T[i])
        maxx = max(testX.T[i])
        rangex = abs(minx) + maxx
        testX.T[i] = testX.T[i] / rangex

    testy = np.append(np.zeros(len(datateb)), np.ones(len(datates)))
    testy = testy.reshape(len(testX), 1)

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

act_fcn = sigmoid
cost_fcn = diff_cost
LR = 0.01
layersize = [len(X[0]), len(X[0])*4, len(X[0])*3, 1]
bias = False

###################################### Train the network #########################################################

# Add bias to X

if bias:
    X = add_bias(X)

# Randomly initialize weights with range -1 to 1

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
        layers[i] = act_fcn(np.dot(layers[i-1], weights[i-1]))

        if bias and not i == NLayers - 1:
            layers[i] = add_bias(layers[i])

    ########### FEED BACKWARDS #########

    lerrors[-1] = cost_fcn(y, layers[-1])
    for i in range(NLayers - 1, 0, -1):

        ldeltas[i] = lerrors[i] * act_fcn(layers[i], deriv=True)

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
    print("\n%i, training set"%i)
    print("Class should have been %i, the ANN guesses %f"%(y[i], layers[-1][i]))

testlayers = [0] * NLayers
testlayers[0] = testX

for i in range(1, NLayers):
    testlayers[i] = act_fcn(np.dot(testlayers[i-1], weights[i-1]))

print("The error for the test set is:" + str(np.mean(np.abs(testy - testlayers[-1]))))
