'''
Bottom-up and Top-down Designs of Privacy Compensation
@Application: Weighted Sum
@Author: Chaoyue Niu
@Email: rvincency@gmail.com
@Reference: C. Niu, Z. Zheng, F. Wu, S. Tang, X. Gao, and G. Chen,
            "Unlocking the Value of Privacy: Trading Aggregate Statistics over Private Correlated Data", in KDD, 2018
@Settings:  ERATO based;
           Total privacy compensations B = 10000.0;
           The dimension of weight vector (numDataOwners) = 1000;
           i: data owner; j: dependent data owners; k: movies.
'''


from random import uniform
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
    Read Dependent Sensitivity
    """
    #NDSS16's Dependent Sensitivity
    #numMovies = 10
    OldDSf = np.zeros((numMovies, numDataOwners), float)
    f1 = open("OldDSf")
    for k in range(numMovies):
        line = f1.readline()
        if not line:
            break
        linesp = line.split()
        for i in range(numDataOwners):
            OldDSf[k][i] = float(linesp[i])
    print(OldDSf)
    f1.close()

    #ERATO's New Dependent Sensitivity
    NewDSf = np.zeros((numMovies, numDataOwners), float)
    f2 = open("NewDSf")
    for k in range(numMovies):
        line = f2.readline()
        if not line:
            break
        linesp = line.split()
        for i in range(numDataOwners):
            NewDSf[k][i] = float(linesp[i])
    print(NewDSf)
    f2.close()

    #Total Privacy Compensations B
    totalMicro = 10000.0

    """
    Bottom-up Design
    OldDSf
    """
    BUMicroPay = np.zeros((numMovies, numDataOwners), float)
    BUMicroCount = np.zeros((numMovies, 200), int)
    BUMicroZeroCount = np.zeros((numMovies,1), int)
    for k in range(numMovies):
        sumOld = 0.0
        for i in range(numDataOwners):
            sumOld += BottomUpCompensate(OldDSf[k][i])
        for i in range(numDataOwners):
            BUMicroPay[k][i] = BottomUpCompensate(OldDSf[k][i])/sumOld * totalMicro
            BUMicroCount[k][int(BUMicroPay[k][i])] += 1
            if(BUMicroPay[k][i] == 0.0):
                BUMicroZeroCount[k] += 1
    np.savetxt('BUMicroPay', BUMicroPay, fmt="%.10f")
    np.savetxt('BUMicroCount', BUMicroCount, fmt="%d")
    np.savetxt('BUMicroZeroCount', BUMicroZeroCount, fmt="%d")

    """
    Top-down Design
    newDSf
    """
    TDMicroPay = np.zeros((numMovies, numDataOwners), float)
    TDMicroCount = np.zeros((numMovies, 200), int)
    TDMicroZeroCount = np.zeros((numMovies, 1), int)
    for k in range(numMovies):
        sumNewDS = 0.0
        for i in range(numDataOwners):
            sumNewDS += NewDSf[k][i]
        for i in range(numDataOwners):
            TDMicroPay[k][i] = NewDSf[k][i]/sumNewDS * totalMicro
            TDMicroCount[k][int(TDMicroPay[k][i])] += 1
            if TDMicroPay[k][i] == 0.0:
                TDMicroZeroCount[k] += 1
    np.savetxt('TDMicroPay', TDMicroPay, fmt = "%.10f")
    np.savetxt('TDMicroCount', TDMicroCount, fmt = "%d")
    np.savetxt('TDMicroZeroCount', TDMicroZeroCount, fmt="%d")
