import numpy
import urllib
import scipy.optimize
import random
"""
def parseData(fname):
  for l in urllib.request.urlopen(fname):
    yield eval(l)

print("Reading data...")
data = list(parseData("http://jmcauley.ucsd.edu/cse190/data/beer/beer_50000.json"))
print("done")

def feature(datum):
  feat = [1]
  return feat

X = [feature(d) for d in data] #得到同樣數量的[1]
y = [d['review/overall'] for d in data]
theta,residuals,rank,s = numpy.linalg.lstsq(X, y)

### Convince ourselves that basic linear algebra operations yield the same answer ###

X = numpy.matrix(X)
y = numpy.matrix(y)
z = numpy.linalg.inv(X.T * X) * X.T * y.T #得到的結果和上面的theta一樣

### Do older people rate beer more highly? ###

data2 = [d for d in data if 'user/ageInSeconds' in d]

def feature(datum):
  feat = [1]
  feat.append(datum['user/ageInSeconds'])
  return feat

X = [feature(d) for d in data2]
y = [d['review/overall'] for d in data2]
theta,residuals,rank,s = numpy.linalg.lstsq(X, y)

### How much do women prefer beer over men? ###

data2 = [d for d in data if 'user/gender' in d]

def feature(datum):
  feat = [1]
  if datum['user/gender'] == "Male":
    feat.append(0)
  else:
    feat.append(1)
  return feat

X = [feature(d) for d in data2]
y = [d['review/overall'] for d in data2]
theta,residuals,rank,s = numpy.linalg.lstsq(X, y)
"""
### Gradient descent ###

# Objective
def f(theta, X, y, lam):
  theta = numpy.matrix(theta).T
  #print(theta)
  X = numpy.matrix(X)
  y = numpy.matrix(y).T
  diff = X*theta - y
  diffSq = diff.T*diff
  diffSqReg = diffSq / len(X) + lam*(theta.T*theta)
  print("offset =", diffSqReg.flatten().tolist())
  return diffSqReg.flatten().tolist()[0]

# Derivative
def fprime(theta, X, y, lam):
  theta = numpy.matrix(theta).T
  #print(theta)
  X = numpy.matrix(X)
  y = numpy.matrix(y).T
  diff = X*theta - y
  res = 2*X.T*diff / len(X) + 2*lam*theta
  #print("gradient =", numpy.array(res.flatten().tolist()[0]))
  return numpy.array(res.flatten().tolist()[0])

X = [[0,5,4,0,0,1,0,0,0,0,1],[0,0,0,6,0,2,0,0,0,0,1]]
y = [5,2]
a = scipy.optimize.fmin_l_bfgs_b(f, [0,0,0,0,0,0,0,0,0,0,0], fprime, args = (X, y, 0.1))

### Random features ###
"""
def feature(datum):
  return [random.random() for x in range(30)]

X = [feature(d) for d in data2]
y = [d['review/overall'] for d in data2]
theta,residuals,rank,s = numpy.linalg.lstsq(X, y)"""