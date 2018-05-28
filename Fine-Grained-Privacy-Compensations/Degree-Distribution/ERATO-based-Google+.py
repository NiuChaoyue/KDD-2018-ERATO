'''
Bottom-up and Top-down Designs of Privacy Compensation
@Application: Degree Distribution (Google+)
@Author: Chaoyue Niu
@Email: rvincency@gmail.com
@Reference: C. Niu, Z. Zheng, F. Wu, S. Tang, X. Gao, and G. Chen,
            "Unlocking the Value of Privacy: Trading Aggregate Statistics over Private Correlated Data", in KDD, 2018
@Settings: ERATO based;
           Total privacy compensations B = 10 * Number of data owners;
           index i: degree + 1.
'''


import numpy as np
import math


def BottomUpCompensate(x):
    v = 10
    insideC = 1.0
    return math.tanh(x/math.sqrt(v/2) * insideC)


if __name__=="__main__":

    maxDegree = 1600

    """
    Load Gplus_Deg_Count
    index i, degree i + 1
    """
    DegCount = np.zeros((maxDegree, 1), int)
    numDataOwners = 0
    f1 = open("Gplus_Deg_Count")
    for i in range(maxDegree):
        line = f1.readline()
        if not line:
            break
        DegCount[i] = int(line)
        numDataOwners += DegCount[i]
    f1.close()
    print(numDataOwners)

    # Total privacy compensations B
    totalMicroPay = numDataOwners * 10.0

    """
    Read Dependent Sensitivities
    """
    totalDSf = 0.0
    DSf = np.zeros((maxDegree, 1), float)
    f2 = open('Google_DSf')
    for i in range(maxDegree):
        line = f2.readline()
        if not line:
            break
        DSf[i] = float(line)
        totalDSf += DSf[i] * DegCount[i]

    """
    Top-down Design
    """
    TDMicroPay = np.zeros((maxDegree, 1), float)
    TDMicroPayCount = np.zeros((1001, 1), int)
    for i in range(maxDegree):
        TDMicroPay[i] = (DSf[i]/totalDSf) * totalMicroPay
        if TDMicroPay[i] < 1000:
            TDMicroPayCount[int(TDMicroPay[i])] += DegCount[i]
        else:
            TDMicroPayCount[1000] += DegCount[i]
    np.savetxt('Google_TD_MicroPayCount', TDMicroPayCount, fmt="%d")

    """
    Bottom-up Design
    """
    BUMicroPay = np.zeros((maxDegree, 1), float)
    BUMicroPayCount = np.zeros((1001, 1), int)
    BUtotalDSf = 0.0
    for i in range(maxDegree):
        BUtotalDSf += BottomUpCompensate(DSf[i]) * DegCount[i]
    for i in range(maxDegree):
        BUMicroPay[i] = (BottomUpCompensate(DSf[i])/BUtotalDSf) * totalMicroPay
        if BUMicroPay[i] < 1000:
            BUMicroPayCount[int(BUMicroPay[i])] += DegCount[i]
        else:
            BUMicroPayCount[1000] += DegCount[i]
    np.savetxt('Google_BU_MicroPayCount', BUMicroPayCount, fmt="%d")
