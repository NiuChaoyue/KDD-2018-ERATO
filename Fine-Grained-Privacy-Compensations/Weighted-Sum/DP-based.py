'''
Bottom-up and Top-down Designs of Privacy Compensation
@Application: Weighted Sum
@Author: Chaoyue Niu
@Email: rvincency@gmail.com
@Reference: C. Niu, Z. Zheng, F. Wu, S. Tang, X. Gao, and G. Chen,
            "Unlocking the Value of Privacy: Trading Aggregate Statistics over Private Correlated Data", in KDD, 2018
@Settings:  Differential Privacy (DP) based;
           Total privacy compensations B = 10000.0;
           The dimension of weight vector (numDataOwners) = 1000;
           i: data owner; k: movies.
'''


import math
import numpy as np


def BottomUpCompensate(x):
    v = 0.1
    insideC = 1.0
    return math.tanh(x/math.sqrt(v/2) * insideC)


if __name__=="__main__":
    numMovies = 10
    numDataOwners = 1000

    """
    Read DP's Sensitivity
    """
    #numMovies = 10
    Sf = np.zeros((numMovies, numDataOwners), float)
    f1 = open("Sf")
    for k in range(numMovies):
        line = f1.readline()
        if not line:
            break
        linesp = line.split()
        for i in range(numDataOwners):
            Sf[k][i] = float(linesp[i])
    print(Sf)
    f1.close()

    #Total Privacy Compensations B
    totalMicro = 10000.0

    """
    Bottom-up Design
    """
    BUMicroPay = np.zeros((numMovies, numDataOwners), float)
    BUMicroCount = np.zeros((numMovies, 200), int)
    BUMicroZeroCount = np.zeros((numMovies,1), int)
    for k in range(numMovies):
        sumBU = 0.0
        for i in range(numDataOwners):
            sumBU += BottomUpCompensate(Sf[k][i])
        for i in range(numDataOwners):
            BUMicroPay[k][i] = BottomUpCompensate(Sf[k][i])/sumBU * totalMicro
            BUMicroCount[k][int(BUMicroPay[k][i])] += 1
            if(BUMicroPay[k][i] == 0.0):
                BUMicroZeroCount[k] += 1
    np.savetxt('DP-BUMicroPay', BUMicroPay, fmt="%.10f")
    np.savetxt('DP-BUMicroCount', BUMicroCount, fmt="%d")
    np.savetxt('DP-BUMicroZeroCount', BUMicroZeroCount, fmt="%d")

    """
    Top-down Design
    """
    TDMicroPay = np.zeros((numMovies, numDataOwners), float)
    TDMicroCount = np.zeros((numMovies, 200), int)
    TDMicroZeroCount = np.zeros((numMovies, 1), int)
    for k in range(numMovies):
        sumTD = 0.0
        for i in range(numDataOwners):
            sumTD += Sf[k][i]
        for i in range(numDataOwners):
            TDMicroPay[k][i] = Sf[k][i]/sumTD * totalMicro
            TDMicroCount[k][int(TDMicroPay[k][i])] += 1
            if TDMicroPay[k][i] == 0.0:
                TDMicroZeroCount[k] += 1
    np.savetxt('DP-TDMicroPay', TDMicroPay, fmt = "%.10f")
    np.savetxt('DP-TDMicroCount', TDMicroCount, fmt = "%d")
    np.savetxt('DP-TDMicroZeroCount', TDMicroZeroCount, fmt="%d")
