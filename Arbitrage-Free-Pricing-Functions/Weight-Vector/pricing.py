'''
Arbitrage-free Pricing Functions
@Part: Weight Vector
       Pricing Function Based on L_p norm and Sigmoid
@Author: Chaoyue Niu
@Email: rvincency@gmail.com
@Reference: C. Niu, Z. Zheng, F. Wu, S. Tang, X. Gao, and G. Chen,
            "Unlocking the Value of Privacy: Trading Aggregate Statistics over Private Correlated Data", in KDD, 2018
@Settings: numDataOwners controls the dimension of weight vector;
           Variance of noise v = 0.1;
           C in tanh(C\pi(S)) = 1.0.
           
'''

import numpy as np
import math
from random import uniform


def paymentFunction(W, num, p, flag):
    #flag; whether apply sigmoid function
    res = 0.0
    v = 0.1
    c = 1.0
    sum = 0.0
    if p != -1:
        for i in range(num):
            sum += math.pow(abs(W[i]), p)
        res = pow(pow(sum, 1.0 / p), 2.0) / v
    else: #max |w_i|
        for i in range(num):
            sum = max(sum, abs(W[i]))
        res = pow(sum, 2.0)/v
    if flag == True:
        res = math.tanh(c * res)
    return res

if __name__=="__main__":
    numDataOwners = 6000
    weightVecs = np.zeros((8, numDataOwners), float)
    file = open("weightVecs_%d"%numDataOwners)
    for i in range(numDataOwners):
        line = file.readline()
        if not line:
            break
        linetmp = line.split()
        for j in range(8):
            weightVecs[j][i] = float(linetmp[j])
    file.close()

    payRes = np.zeros((8,1), float)
    """
    L_1, L_2, L_3, and L_Inf norms
    """
    for j in range(8):
        payRes[0] += paymentFunction(weightVecs[j], numDataOwners, 1, False)/8.0
        payRes[1] += paymentFunction(weightVecs[j], numDataOwners, 2, False) / 8.0
        payRes[2] += paymentFunction(weightVecs[j], numDataOwners, 3, False) / 8.0
        payRes[3] += paymentFunction(weightVecs[j], numDataOwners, -1, False) / 8.0
        payRes[4] += paymentFunction(weightVecs[j], numDataOwners, 1, True) / 8.0
        payRes[5] += paymentFunction(weightVecs[j], numDataOwners, 2, True) / 8.0
        payRes[6] += paymentFunction(weightVecs[j], numDataOwners, 3, True) / 8.0
        payRes[7] += paymentFunction(weightVecs[j], numDataOwners, -1, True) / 8.0
    print(payRes)
    np.savetxt("PayRes_%d" % numDataOwners, payRes, fmt='%.10f')

