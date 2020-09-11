import numpy as np 
import math
from matplotlib import pyplot as plt

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
    x_train, y_train_corrupt, x_test, y_test = corrupt_sine(10, 0.1, 0, 1)
    SSEs = [0]*16
    for i in range(0,16):
        p= poly_fitting(i, x_train, y_train_corrupt)
        # plt.plot(train_x, PolyCoefficients(train_x, p))
        y_predict = predict(x_test,p)

        # plt.show()
        SSEs[i] = SSE(10, y_test,y_predict)
    M_list = list(range(0,16))
    plt.title("part_2: SSE(y_axis) v/s M(x_axis)")
    plt.plot(M_list, SSEs)
    plt.show()

    return

def part_3():
    N_list = list(range(10,1001))
    # print(len(N_list))
    SSEs = [0]*(1001-10)
    # print(len(SSEs))
    for i in range(0,len(N_list)):
        x_train, y_train_corrupt, x_test, y_test = corrupt_sine(N_list[i], 0.1, 0, 1)
        p = poly_fitting(9, x_train, y_train_corrupt)
        y_predict = predict(x_test, p)
        SSEs[i] = SSE(N_list[i],y_test,y_predict)
    plt.title("part_3: SSE(y_axis) v/s N(x_axis)")
    plt.plot(N_list, SSEs)
    plt.show()

    return

def part_4():
    # gammas = list(range(0.1:0.01:0.51))
    gammas = [i for i in np.arange(0.1,0.51,0.01)]
    SSEs = [0]*len(gammas)
    for i in range(0, len(gammas)):
        x_train, y_train_corrupt, x_test, y_test = corrupt_sine(100, gammas[i], 0, 1)
        p = poly_fitting(10, x_train, y_train_corrupt)
        y_predict = predict(x_test, p)
        SSEs[i] = SSE(100,y_test,y_predict)
    plt.title("part_3: SSE(y_axis) v/s gamma(x_axis)")
    plt.plot(gammas, SSEs)
    plt.show()


def main():
    # part_2()
    # part_3()
    # part_4()

if __name__=="__main__":
    main()
