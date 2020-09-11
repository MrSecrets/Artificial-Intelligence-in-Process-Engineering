import numpy as np 
import math
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge 



# toy = np.array([0, 2, 5])
# PolynomialFeatures(4).fit_transform(toy.reshape(-1,1))



# d = 20 # Maximum polynomial degree
# You will create a grid of plots of this size (7 x 2)
# rows = 7
# # cols = 2
# lambdas = [0., 1e-6, 1e-3, 1e-2, 1e-1, 1, 10] # Various penalization parameters to try
# grid_to_predict = np.arange(0, 1, .01) # Predictions will be made on this grid

# # Create training set and test set
# # Xtrain = PolynomialFeatures(d).fit_transform(xtrain.reshape(-1,1))
# # test_set = PolynomialFeatures(d).fit_transform(grid_to_predict.reshape(-1,1))

# # fig, axs = plt.subplots(rows, cols, sharex='col', figsize=(12, 24)) # Set up plotting objects

# for i, lam in enumerate(lambdas):
#     # your code here
#     ridge_reg = Ridge(alpha = lam) # Create regression object
#     ridge_reg.fit(Xtrain, ytrain) # Fit on regression object
#     ypredict_ridge = ridge_reg.predict(test_set) # Do a prediction on the test set
    
#     ### Provided code
#     axs[i,0].plot(xtrain, ytrain, 's', alpha=0.4, ms=10, label="in-sample y") # Plot sample observations
#     axs[i,0].plot(grid_to_predict, ypredict_ridge, 'k-', label=r"$\lambda =  {0}$".format(lam)) # Ridge regression prediction
#     axs[i,0].set_ylabel('$y$') # y axis label
#     axs[i,0].set_ylim((0, 1)) # y axis limits
#     axs[i,0].set_xlim((0, 1)) # x axis limits
#     axs[i,0].legend(loc='best') # legend
    
#     coef = ridge_reg.coef_.ravel() # Unpack the coefficients from the regression
    
#     axs[i,1].semilogy(np.abs(coef), ls=' ', marker='o', label=r"$\lambda =  {0}$".format(lam)) # plot coefficients
#     axs[i,1].set_ylim((1e-04, 1e+15)) # Set y axis limits
#     axs[i,1].set_xlim(1, 20) # Set y axis limits
#     axs[i,1].yaxis.set_label_position("right") # Move y-axis label to right
#     axs[i,1].set_ylabel(r'$\left|\beta_{j}\right|$') # Label y-axis
#     axs[i,1].legend(loc='best') # Legend

# # Label x axes
# axs[-1, 0].set_xlabel("x")
# axs[-1, 1].set_xlabel(r"$j$");








def PolyCoefficients(x, coeffs):
    """ Returns a polynomial for ``x`` values for the ``coeffs`` provided.

    The coefficients must be in ascending order (``x**0`` to ``x**o``).
    """
    o = len(coeffs)
    # print(f'# This is a polynomial of order {ord}.')
    y = 0
    for i in range(o):
        y += coeffs[i]*x**i
    return y

def predict(x, coeffs):
	y_predict = [0]*len(x)
	for i in range(0,len(x)):
		for j in range(len(coeffs)):
			y_predict[i] += coeffs[j]*x[i]**j

	return y_predict

def corrupt_sine(N, gamma, loc, scale):
	noise = gamma*np.random.normal(loc=loc, scale=scale)
	x_train = np.random.uniform(low=0, high=1, size = N)
	y_error = np.sin(2*math.pi*x_train) + noise
	x_test = np.random.uniform(low=0, high=1, size = N)
	y_test = np.sin(2*math.pi*x_test) + noise

	return x_train,y_error,x_test,y_test

def poly_fitting(M, x, y):
    p = np.polyfit(x=x, y=y, deg=M)
    p = np.flipud(p)
    return p

def SSE(N, y_test,y_predict):
    square_error = 0
    for i in range(0,N):
        square_error += (y_predict[i]-y_test[i])**2

    # print(np.sqrt(abs(square_error)), "    888888888888888888888888")
    return np.sqrt(abs(square_error))

def part_2():
    lam = 0.5
    x_train, y_train_corrupt, x_test, y_test = corrupt_sine(10, 0.1, 0, 1)
    Xtrain = PolynomialFeatures(10).fit_transform(x_train.reshape(-1,1))
    test_set = PolynomialFeatures(10).fit_transform(x_test.reshape(-1,1))
    ridge_reg = Ridge(alpha = lam) # Create regression object
    ridge_reg.fit(Xtrain, y_train_corrupt) # Fit on regression object
    ypredict_ridge = ridge_reg.predict(test_set)


    SSEs = [0]*16
    for i in range(0,16):
        Xtrain = PolynomialFeatures(i).fit_transform(x_train.reshape(-1,1))
        test_set = PolynomialFeatures(i).fit_transform(x_test.reshape(-1,1))
        ridge_reg = Ridge(alpha = lam) # Create regression object
        ridge_reg.fit(Xtrain, y_train_corrupt) # Fit on regression object
        ypredict_ridge = ridge_reg.predict(test_set)
        # y_predict = predict(x_test,p)
        SSEs[i] = SSE(10, y_test,ypredict_ridge)
    M_list = list(range(0,16))
    plt.title("part_2: SSE(y_axis) v/s M(x_axis)   lambda = 0.5")
    plt.plot(M_list, SSEs)
    plt.show()

    return

def part_3():
    N_list = list(range(10,1001))
    # print(len(N_list))
    SSEs = [0]*(1001-10)
    lam = 0.5

    # print(len(SSEs))
    for i in range(0,len(N_list)):
        print(i)
        x_train, y_train_corrupt, x_test, y_test = corrupt_sine(N_list[i], 0.1, 0, 1)
        Xtrain = PolynomialFeatures(9).fit_transform(x_train.reshape(-1,1))
        test_set = PolynomialFeatures(9).fit_transform(x_test.reshape(-1,1))
        ridge_reg = Ridge(alpha = lam) # Create regression object
        ridge_reg.fit(Xtrain, y_train_corrupt) # Fit on regression object
        ypredict_ridge = ridge_reg.predict(test_set)
        SSEs[i] = SSE(N_list[i],y_test,ypredict_ridge)
    plt.title("part_3: SSE(y_axis) v/s N(x_axis)   lambda = 0.5")
    plt.plot(N_list, SSEs)
    plt.show()

    return

def part_4():
    # gammas = list(range(0.1:0.01:0.51))
    gammas = [i for i in np.arange(0.1,0.51,0.01)]
    lam = 0.5
    SSEs = [0]*len(gammas)
    for i in range(0, len(gammas)):
        x_train, y_train_corrupt, x_test, y_test = corrupt_sine(100, gammas[i], 0, 1)
        Xtrain = PolynomialFeatures(9).fit_transform(x_train.reshape(-1,1))
        test_set = PolynomialFeatures(9).fit_transform(x_test.reshape(-1,1))
        ridge_reg = Ridge(alpha = lam) # Create regression object
        ridge_reg.fit(Xtrain, y_train_corrupt) # Fit on regression object
        ypredict_ridge = ridge_reg.predict(test_set)
        # p = poly_fitting(10, x_train, y_train_corrupt)
        # y_predict = predict(x_test, p)
        SSEs[i] = SSE(100,y_test,ypredict_ridge)
    plt.title("part_3: SSE(y_axis) v/s gamma(x_axis)   lambda = 0.5")
    plt.plot(gammas, SSEs)
    plt.show()


def main():
    part_2()
    part_3()
    part_4()

if __name__=="__main__":
    main()
