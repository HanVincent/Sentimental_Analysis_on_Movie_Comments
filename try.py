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
    for indexDict in range(len(X)):        
        temp = 0 #store the i,jth number
        for indexVector in range(len(theta)):
            if indexVector in X[indexDict]:
                temp += X[indexDict][indexVector] * theta[indexVector]
            elif indexVector not in X[indexDict]:
                temp += 0
            else:
                print("ERROR!1")
                
        diff.append(temp - y[indexDict])

    # diffSq = diff.T * diff
    diff = numpy.matrix(diff)    
    diffSq = diff * diff.T
    theta = numpy.matrix(theta)
    diffSqReg = diffSq / len(X) + lam * (theta * theta.T)
    print("offset =", diffSqReg.flatten().tolist())
    return diffSqReg.flatten().tolist()[0]  
  
### Derivative 算完成
def fprime(theta, X, y, lam): 
    len_X = len(X)
    len_theta = len(theta)
    
    # diff = X*theta - y
    diff = []
    for indexDict in range(len_X):        
        temp = 0 #store the i,jth number
        for indexVector in range(len_theta):
            if indexVector in X[indexDict]:
                temp += X[indexDict][indexVector] * theta[indexVector]
            elif indexVector not in X[indexDict]:
                temp += 0
            else:
                print("ERROR!1")
                
        diff.append(temp - y[indexDict])

    # res = 2*X.T*diff / len(X) + 2*lam*theta
    res = []
    for indexVector in range(len_theta):
        temp = 0
        for indexDict in range(len_X):
            if indexVector in X[indexDict]:
                temp += X[indexDict][indexVector] * diff[indexDict]
            elif indexVector not in X[indexDict]:
                temp += 0
            else:
                print("ERROR!2")
        
        temp = 2 * temp / len_X + 2 * lam * theta[indexVector]
        res.append(temp)

    res = numpy.matrix(res)
    print("gradient =", numpy.array(res.flatten().tolist()[0]))
    return numpy.array(res.flatten().tolist()[0])
  
### Main
#num = 11 #1000, 2000...
#X = [{1:5, 2:4, 5:1, 10:1},{5:2, 3:6, 10:1}]
#y = [5,2]
#theta = [0] * num
#f(theta, X, y, num)

X = [{1:5, 2:4, 5:1, 10:1},{5:2, 3:6, 10:1}]
y = [5,2]
a = scipy.optimize.fmin_l_bfgs_b(f, [0,0,0,0,0,0,0,0,0,0,0], fprime, args = (X, y, 0.1))
