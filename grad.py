import numpy
import scipy.optimize
"""
def Gradient(theta, X, y, lam):
    temp_theta = theta

    # theta = theta - alpha * f'(theta)
    while True: #abs(x_new - x_old) > precision  
        temp_theta = theta
        theta = theta - alpha * fprime(theta, X, y, lam = 0.1)
        break
"""
# Objective
def f(theta, X, y, lam):
    # diff = X*theta - y
    diff = []
    for index, eachDict in enumerate(X):
        temp = 0 #store the i,jth number
        for k, v in eachDict.items():
            temp += v * theta[k]
            
        diff.append(temp - y[index])
        
    # diffSq = diff.T * diff
    diff = numpy.matrix(diff)    
    diffSq = diff * diff.T
    theta = numpy.matrix(theta)
    diffSqReg = diffSq / len(X) + lam * (theta * theta.T)
    #print("offset =", diffSqReg.flatten().tolist())
    return diffSqReg.flatten().tolist()[0]  
  
### Derivative 
def fprime(theta, X, y, lam): 
    # diff = X*theta - y
    diff = []
    for index, eachDict in enumerate(X):
        temp = 0
        for k, v in eachDict.items():
            temp += v * theta[k]
            
        diff.append(temp - y[index])

    # res = 2*X.T*diff / len(X) + 2*lam*theta
    len_X = len(X)
    res = [0] * len(theta)
    for index, eachDict in enumerate(X):
        for k, v in eachDict.items():
            res[k] += 2 * v * diff[index] / len_X
    
    res = numpy.matrix(res)
    res = res + 2 * lam * theta 
    
    #print("gradient =", numpy.array(res.flatten().tolist()[0]))
    return numpy.array(res.flatten().tolist()[0])

### Predict
def predict(test_X, theta):
    y = [0] * len(test_X)
    for index, eachDict in enumerate(test_X):
        for k, v in eachDict.items():
            y[index] += v * theta[k]
    
    return(y)    

### Main Test
#num = 11 #1000, 2000...
#X = [{1:5, 2:4, 5:1, 10:1},{5:2, 3:6, 10:1}]
#y = [5,2]
#theta = [0] * num
#f(theta, X, y, num)
#
#X = [{1:5, 2:4, 5:1, 10:1},{5:2, 3:6, 10:1}]
#y = [5,2]
#a = scipy.optimize.fmin_l_bfgs_b(f, [0,0,0,0,0,0,0,0,0,0,0], fprime, args = (X, y, 0.1))
