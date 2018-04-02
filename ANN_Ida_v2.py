import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def kdeplot2(dat1, dat2, par=0.37):
    mini = min([min(dat1), min(dat2)])
    maxi = max([max(dat1), max(dat2)])
    xl = np.linspace(mini, maxi, num=500)
    kdeobj1 = gaussian_kde(dat1, bw_method=par)
    kde1 = kdeobj1.evaluate(xl)
    kdeobj2 = gaussian_kde(dat2, bw_method=par)
    kde2 = kdeobj2.evaluate(xl)

    # plt.figure()
    f, ax = plt.subplots()
    ax.plot(xl, kde1)
    ax.plot(xl, kde2)

    return ax


def sigmoid(x, deriv=False):
    if deriv:
        return x * (1 - x)  # This x is actually sigmoid(x), not the ind var x

    return 1 / (1 + np.exp(-x))


def relu(x, deriv=False):
    if type(x) == list:
        x = np.array(x)
    if deriv:
        # outlist = []
        # xflat = x.flatten()
        # for xp in xflat:
        #     if xp > 0:
        #         outlist.append(1)
        #     else:
        #         outlist.append(0)
        # outlist = np.array(outlist)
        # outlist = outlist.reshape(x.shape)
        outlist = x > 0
        outlist = outlist.astype(int)
        return outlist
    # outlist = []
    # xflat = x.flatten()
    # for xp in xflat:
    #     if xp > 0:
    #         outlist.append(xp)
    #     else:
    #         outlist.append(0)
    # outlist = np.array(outlist)
    # outlist = outlist.reshape(x.shape)
    outlist = x.clip(min=0)
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


def cross_ent_cost(yl, lastlyr):
    yl = yl.T
    lastlyr = lastlyr.T
    return -(1.0 / len(X)) * (np.dot(np.log(lastlyr), yl.T) + np.dot(np.log(1 - lastlyr), (1 - yl).T))


def quad_cost(yl, lastlyr):
    return (yl - lastlyr)**2


def cross_ent_cost_p(yl, lastlyr):
    lastlyr -= 0.00000001
    cost = (yl - lastlyr) / ((1 - lastlyr) * lastlyr)  # Is the sign correct?
    return cost


def quad_cost_p(yl, lastlyr):
    return yl - lastlyr


###################################### Import data ##############################################

useadvBDTdat = True
usesimBDTdat = False

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

    testX = np.array(datatest)

    for i in range(len(testX[0])):
        minx = min(testX.T[i])
        maxx = max(testX.T[i])
        rangex = abs(minx) + maxx
        testX.T[i] = testX.T[i] / rangex

    Ndhalft = len(datatest) / 2
    testy = [0, 1] * int(Ndhalft)
    testy = np.array(testy)
    testy = testy.reshape(len(datatest), 1)

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
                 [0, 1, 1, 0],
                 [1, 1, 1, 0],
                 [0, 0, 1, 1]])

    y = np.array([[1],
                  [1],
                  [1],
                  [0],
                  [0],
                  [0]])

    testX = np.array([[1, 0, 1, 0],
                      [1, 0, 1, 1],
                      [0, 0, 1, 0],
                      [1, 0, 0, 1]])

    testy = np.array([[0],
                      [0],
                      [1],
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
cost_fcn_p = cross_ent_cost_p
cost_fcn = cross_ent_cost
LR = 10**(-4)
Nit = 20000
Nprint = 100
layersize = [len(X[0]), len(X[0])*5, len(X[0])*5, 1]
bias = False

print("\n" + "Current Settings: LR = %f, NLayers = %i"%(LR, len(layersize)))


# Set #1 of settings for BDTsimdat

# act_fcn = sigmoid
# cost_fcn_p = quad_cost_p
# LR = 0.01
# Nit = 15000
# Nprint = 1000
# layersize = [len(X[0]), len(X[0])*4, len(X[0])*3, 1]
# bias = False


# Set #2 of settings for BDTsimdat

# act_fcn = sigmoid
# cost_fcn_p = cross_ent_cost_p
# LR = 10**(-4)
# Nit = 15000
# Nprint = 1000
# layersize = [len(X[0]), len(X[0])*4, len(X[0])*3, 1]
# bias = False


#  Set #1 of settings for BDTadvdat

# act_fcn = sigmoid
# cost_fcn_p = cross_ent_cost_p
# cost_fcn = cross_ent_cost
# LR = 10**(-4)
# Nit = 20000
# Nprint = 100
# layersize = [len(X[0]), len(X[0])*5, len(X[0])*5, 1]
# bias = False

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

cost_list = []
cost_p_list = []
j_comp = []

layers = [0] * NLayers
lerrors = [0] * NLayers
ldeltas = [0] * NLayers

for j in range(Nit + 1):

    ############ FEED FORWARD ###########

    layers[0] = X
    for i in range(1, NLayers):
        layers[i] = act_fcn(np.dot(layers[i-1], weights[i-1]))

        if bias and not i == NLayers - 1:
            layers[i] = add_bias(layers[i])

    ########### FEED BACKWARDS #########

    lerrors[-1] = cost_fcn_p(y, layers[-1])
    for i in range(NLayers - 1, 0, -1):

        ldeltas[i] = lerrors[i] * act_fcn(layers[i], deriv=True)

        if bias and i < NLayers - 1:
            ldeltasn = ldeltas[i][:, 1:]
            lerrors[i - 1] = ldeltasn.dot(weights[i - 1].T)
            weights[i - 1] += layers[i - 1].T.dot(ldeltasn) * LR

        else:
            lerrors[i - 1] = ldeltas[i].dot(weights[i - 1].T)
            weights[i - 1] += layers[i - 1].T.dot(ldeltas[i]) * LR

    if j % Nprint == 0:
        cost_val = np.sum(cost_fcn(y, layers[-1]))
        cost_p_val = np.mean(np.abs(lerrors[-1]))
        print("\nj is %i out of %i"%(j, Nit))
        print("The derivative of the cost is currently %f. The derivative of the quad "
              "cost is %f."%(cost_p_val, np.mean(np.abs(quad_cost_p(y, layers[-1])))))
        print("The cost has been decreased to %f. The quad "
              "cost is %f." % (cost_val, np.sum(quad_cost(y, layers[-1]))))
        j_comp.append(j)
        cost_list.append(cost_val)
        cost_p_list.append(cost_p_val)

# Plot of the cost function and it's d

plt.figure()
plt.plot(j_comp, cost_list, '-*')
plt.xlabel("Number of iterations")
plt.ylabel("Cost function")
plt.savefig("Cost_plot.pdf")

plt.figure()
plt.plot(j_comp, cost_p_list, '-*')
plt.xlabel("Number of iterations")
plt.ylabel("Derivative of the cost function")
plt.savefig("Cost_p_plot.pdf")

###################################### Test the network #########################################################

for i in range(5):
    print("\n" + "%i, training set"%i)
    print("Class should have been %i, the ANN guesses %f"%(y[i], layers[-1][i]))

testlayers = [0] * NLayers
testlayers[0] = testX

for i in range(1, NLayers):
    testlayers[i] = act_fcn(np.dot(testlayers[i-1], weights[i-1]))

print("\n" + "The cost for the test set is %f. The quad cost for the test set is "
      "%f." % (np.sum(cost_fcn(testy, testlayers[-1])), np.sum(quad_cost(testy, testlayers[-1]))))
print("The linear error for the test set is: " + str(np.mean(np.abs(testy - testlayers[-1]))))


###################################### Histograms #########################################################

# Training set

plt.figure()
binwidth = 0.04

i_1, cake = np.where(y == 1)  # The cake is a lie, don't use it

# y_pred_s = [x for i, x in enumerate(layers[-1]) if y[i]==1]  # predicted y for real signal
# y_pred_b = [x for i, x in enumerate(layers[-1]) if y[i]==0]  # predicted y for real background

y_pred_s = []  # predicted y for real signal
y_pred_b = []  # predicted y for real background


for i, x in enumerate(layers[-1]):
    if y[i] == 1:
        y_pred_s = np.append(y_pred_s, x)
    else:
        y_pred_b = np.append(y_pred_b, x)
    # if i % 500 == 0:
    #     print(i, " of ", len(layers[-1]))

binwidth = 0.05
plt.hist(y_pred_s, 20, range=[0, 1], normed=0, facecolor='red', alpha=0.5)
# counts, bins, # patches = #bins=range(min(y_pred_s), max(y_pred_s) + binwidth, binwidth)
plt.hist(y_pred_b, 20,range=[0, 1], normed=0, facecolor='blue', alpha=0.5)
plt.xlabel(' Output layer')
plt.ylabel(' Frequency')

plt.legend(["Signal", "Background"], loc='upper center')
plt.title("Training set", fontsize=16)

# plt.xlim(0,1)
# plt.ylim(0,4)

dat1 = y_pred_s
dat2 = y_pred_b
par = 0.37

mini = min([min(dat1), min(dat2)])
maxi = max([max(dat1), max(dat2)])
xl = np.linspace(mini, maxi, num=500)
kdeobj1 = gaussian_kde(dat1, bw_method=par)
kde1 = kdeobj1.evaluate(xl)
kdeobj2 = gaussian_kde(dat2, bw_method=par)
kde2 = kdeobj2.evaluate(xl)

plt.figure()
plt.plot(xl, kde1, color='red')
plt.plot(xl, kde2, color='blue')

plt.xlabel(' Output layer')
plt.ylabel(' Frequency')

plt.legend(["Signal", "Background"], loc='upper center')
plt.title("Training set", fontsize=16)


# Test set

plt.figure()
binwidth = 0.04

i_1, cake = np.where(testy == 1)  # The cake is a lie, don't use it

# y_pred_s = [x for i, x in enumerate(layers[-1]) if y[i]==1]  # predicted y for real signal
# y_pred_b = [x for i, x in enumerate(layers[-1]) if y[i]==0]  # predicted y for real background

y_pred_s = []  # predicted y for real signal
y_pred_b = []  # predicted y for real background


for i, x in enumerate(testlayers[-1]):
    if testy[i] == 1:
        y_pred_s = np.append(y_pred_s, x)
    else:
        y_pred_b = np.append(y_pred_b, x)
    # if i % 500 == 0:
    #     print(i, " of ", len(testlayers[-1]))

binwidth = 0.05
plt.hist(y_pred_s, 20, range=[0, 1], normed=0, facecolor='red', alpha=0.5)
# counts, bins, # patches = #bins=range(min(y_pred_s), max(y_pred_s) + binwidth, binwidth)
plt.hist(y_pred_b, 20,range=[0, 1], normed=0, facecolor='blue', alpha=0.5)
plt.xlabel(' Output layer')
plt.ylabel(' Frequency')

plt.legend(["Signal", "Background"], loc='upper center')
plt.title("Test set", fontsize=16)

dat1 = y_pred_s
dat2 = y_pred_b
par = 0.37

mini = min([min(dat1), min(dat2)])
maxi = max([max(dat1), max(dat2)])
xl = np.linspace(mini, maxi, num=500)
kdeobj1 = gaussian_kde(dat1, bw_method=par)
kde1 = kdeobj1.evaluate(xl)
kdeobj2 = gaussian_kde(dat2, bw_method=par)
kde2 = kdeobj2.evaluate(xl)

plt.figure()
plt.plot(xl, kde1, color='red')
plt.plot(xl, kde2, color='blue')

plt.xlabel(' Output layer')
plt.ylabel(' Frequency')

plt.legend(["Signal", "Background"], loc='upper center')
plt.title("Test set", fontsize=16)

plt.show()
