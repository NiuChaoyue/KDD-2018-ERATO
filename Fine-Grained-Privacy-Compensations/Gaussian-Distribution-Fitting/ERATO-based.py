'''
Bottom-up and Top-down Designs of Privacy Compensation
@Application: Gaussian Distribution Fitting (Sum and Sum of Squares)
@Author: Chaoyue Niu
@Email: rvincency@gmail.com
@Reference: C. Niu, Z. Zheng, F. Wu, S. Tang, X. Gao, and G. Chen,
            "Unlocking the Value of Privacy: Trading Aggregate Statistics over Private Correlated Data", in KDD, 2018
@Settings: ERATO based;
           Total privacy compensations B = 100000.0;
           Number of data owners = 10000;
           i: data owner; j: dependent data owners; k: type of energy consumption.
'''


from random import uniform
import math
import numpy as np


def xBottomUpCompensate(x, k):
    v = 100.0
    return math.tanh(x/math.sqrt(v/2))


def x2BottomUpCompensate(x):
    v = 100.0
    return math.tanh(x/math.sqrt(v/2))


if __name__=="__main__":
    """
    Some Prior Information about MIN, MAX
    """
    #For Sum: E(x)
    lowBound = [0, 0, 0, 0, 0]
    upBound = [5000, 3000, 3000, 400, 5000]


    """
    Read Dependent Sensitivity
    """
    #For Sum: E(x)
    xDSf = np.zeros((10000,5),float)
    f1 = open("xDSf")
    for i in range(10000):
        line = f1.readline()
        if not line:
            break
        linetmp = line.split()
        for k in range(5):
            xDSf[i][k] = float(linetmp[k])
    print(xDSf)
    f1.close()

    # For Sum of Squares: E(x^2)
    x2DSf = np.zeros((10000, 5), float)
    f2 = open("x2DSf")
    for i in range(10000):
        line = f2.readline()
        if not line:
            break
        linetmp = line.split()
        for k in range(5):
            x2DSf[i][k] = float(linetmp[k])
    print(x2DSf)
    f2.close()

    #Total Privacy Compensations B
    total_micro = 100000.0


    """
    Case of Sum: E(x) 
    """
    #Bottom-up Design
    xbuPsi = np.zeros((10000, 5), float)
    xbusumPsi = np.zeros((5, 1), float)
    for k in range(5):
        for i in range(10000):
            xbuPsi[i][k] = xBottomUpCompensate(xDSf[i][k], k)
            xbusumPsi[k] += xbuPsi[i][k]

    xbuMicroPay = np.zeros((10000, 5), float)
    # xbuCnt: Count different interval values
    xbuCnt = np.zeros((6, 50), int)
    xbuMicoPayAvg = np.zeros((10000, 1), float)
    for k in range(5):
        for i in range(10000):
            xbuMicroPay[i][k] = (xbuPsi[i][k]/xbusumPsi[k]) * total_micro
            xbuMicoPayAvg[i] += xbuMicroPay[i][k]/5.0
            xbuCnt[k][int(xbuMicroPay[i][k])] += 1
    for i in range(10000):
        xbuCnt[5][int(xbuMicoPayAvg[i])] += 1
    np.savetxt('xbuMicroPay', xbuMicroPay, fmt='%.10f')
    np.savetxt('xbuMicroPay-Count', xbuCnt, fmt='%d')

    #Top-down Design
    xtdsumDSfi = np.zeros((5, 1), float)
    for k in range(5):
        for i in range(10000):
            xtdsumDSfi[k] += xDSf[i][k]

    xtdMicroPay = np.zeros((10000, 5), float)
    # xtdCnt: Count different interval values
    xtdCnt = np.zeros((6, 50), int)
    xtdMicoPayAvg = np.zeros((10000, 1), float)

    for k in range(5):
        for i in range(10000):
            xtdMicroPay[i][k] = (xDSf[i][k] / xtdsumDSfi[k]) * total_micro
            xtdMicoPayAvg[i] += xtdMicroPay[i][k] / 5.0
            xtdCnt[k][int(xtdMicroPay[i][k])] += 1
    for i in range(10000):
        xtdCnt[5][int(xtdMicoPayAvg[i])] += 1
    np.savetxt('xtdMicroPay', xtdMicroPay, fmt='%.10f')
    np.savetxt('xtdMicroPay-Count', xtdCnt, fmt='%d')

    """
    Case of Sum of Squares: E(x^2)
    """
    # Bottom-up Design
    x2buPsi = np.zeros((10000, 5), float)
    x2busumPsi = np.zeros((5, 1), float)
    for k in range(5):
        for i in range(10000):
            x2buPsi[i][k] = x2BottomUpCompensate(x2DSf[i][k])
            # print(buPsi[i][k])
            x2busumPsi[k] += x2buPsi[i][k]

    x2buMicroPay = np.zeros((10000, 5), float)
    # x2buCnt: Count different interval values
    x2buCnt = np.zeros((6, 100), int)
    x2buMicoPayAvg = np.zeros((10000, 1), float)
    for k in range(5):
        for i in range(10000):
            x2buMicroPay[i][k] = (x2buPsi[i][k] / x2busumPsi[k]) * total_micro
            x2buMicoPayAvg[i] += x2buMicroPay[i][k] / 5.0
            x2buCnt[k][int(x2buMicroPay[i][k])] += 1
    for i in range(10000):
        x2buCnt[5][int(x2buMicoPayAvg[i])] += 1
    np.savetxt('x2buMicroPay', x2buMicroPay, fmt='%.10f')
    np.savetxt('x2buMicroPay-Count', x2buCnt, fmt='%d')

    # Top-down Design
    x2tdsumDSfi = np.zeros((5, 1), float)
    for k in range(5):
        for i in range(10000):
            x2tdsumDSfi[k] += x2DSf[i][k]

    x2tdMicroPay = np.zeros((10000, 5), float)
    # x2tdCnt: Count different interval values
    x2tdCnt = np.zeros((6, 100), int)
    x2tdMicoPayAvg = np.zeros((10000, 1), float)

    for k in range(5):
        for i in range(10000):
            x2tdMicroPay[i][k] = (x2DSf[i][k]/ x2tdsumDSfi[k]) * total_micro
            x2tdMicoPayAvg[i] += x2tdMicroPay[i][k] / 5.0
            x2tdCnt[k][int(x2tdMicroPay[i][k])] += 1
    for i in range(10000):
        x2tdCnt[5][int(x2tdMicoPayAvg[i])] += 1
    np.savetxt('x2tdMicroPay', x2tdMicroPay, fmt='%.10f')
    np.savetxt('x2tdMicroPay-Count', x2tdCnt, fmt='%d')
