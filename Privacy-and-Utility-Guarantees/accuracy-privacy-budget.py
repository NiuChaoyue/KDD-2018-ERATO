'''
Privacy and Utility Guarantees
@Application: Weighted Sum
@Author: Chaoyue Niu
@Email: rvincency@gmail.com
@Reference: C. Niu, Z. Zheng, F. Wu, S. Tang, X. Gao, and G. Chen,
            "Unlocking the Value of Privacy: Trading Aggregate Statistics over Private Correlated Data", in KDD, 2018
@Settings: Perturbation Mechanisms: ERATO, Differential Privacy (DP), and Dependent Differential Privacy (DDP)
           indexNum controls the privacy budget;
           The dimension of weight vector (numDataOwners) = 1000;
           i: data owner; j: dependent owners; k: movies.
'''


from random import uniform
import math
import numpy as np


def Lap_noise(loc, sensitivity, epsilon):
    scale = sensitivity / epsilon
    num = 10000
    samples = np.random.laplace(loc, scale, num)
    sumS = 0.0
    for i in range(num):
        sumS += abs(samples[i])
    return sumS/num


if __name__=="__main__":
    """
    Set Variables
    """
    indexNum = -1
    epsilon = math.pow(10,indexNum)
    numMovies = 10
    numDataOwners = 1000

    """
    Read Sensitivities
    """
    #Differential Privacy's Sensitivity
    Sf = np.zeros((numMovies, numDataOwners), float)
    f0 = open("Sf")
    for k in range(numMovies):
        line = f0.readline()
        if not line:
            break
        linesp = line.split()
        for i in range(numDataOwners):
            Sf[k][i] = float(linesp[i])
    f0.close()

    #NDSS16's Dependent sensitivity
    OldDSf = np.zeros((numMovies, numDataOwners), float)
    f1 = open("OldDSf")
    for k in range(numMovies):
        line = f1.readline()
        if not line:
            break
        linesp = line.split()
        for i in range(numDataOwners):
            OldDSf[k][i] = float(linesp[i])
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
    f2.close()

    """
    Read Movie Data, default 10 x 1000
    """
    f3 = open("Movies%dx%d" % (numMovies, numDataOwners))
    rawdata = np.zeros((numMovies, numDataOwners), float)
    for k in range(numMovies):
        line = f3.readline()
        if not line:
            break
        linetmp = line.split()
        for i in range(numDataOwners):
            rawdata[k][i] = float(linetmp[i])
    f3.close()

    """
    Read Weight Vector,  default movie 0
    """
    weightVecs = np.zeros((1, numDataOwners), float)
    f4 = open("weightVecs_%d" % numDataOwners)
    for i in range(numDataOwners):
        line = f4.readline()
        if not line:
            break
        linetmp = line.split()
        weightVecs[0][i] = float(linetmp[3])
    f4.close()
    intercept = 2.5556573744


    """
    True Results
    """
    TrueRes = np.zeros((numMovies, 1), float)
    for k in range(numMovies):
        TrueRes[k] = np.dot(weightVecs[0], rawdata[k]) + intercept
    print("Ture Results:\n", TrueRes)
    np.savetxt("TrueRes", TrueRes, fmt='%.10f')

    MaxSFTable = np.zeros((3, numMovies), float)

    """
    Differential Privacy
    """
    DPRes = np.zeros((numMovies, 1), float)
    for k in range(numMovies):
        #find max sensitvity'
        maxsf = 0.0
        for i in range(numDataOwners):
            maxsf = max(maxsf, Sf[k][i])
        MaxSFTable[0][k] = maxsf
        DPRes[k] = TrueRes[k]
        DPRes[k] += Lap_noise(0, maxsf, epsilon)

    """
    NDSS'16 Depedenet Differential Privacy
    """
    DDPRes = np.zeros((numMovies, 1), float)
    for k in range(numMovies):
        maxDSf = 0.0
        for i in range(numDataOwners):
            maxDSf = max(maxDSf, OldDSf[k][i])
        MaxSFTable[1][k] = maxDSf
        DDPRes[k] = TrueRes[k]
        DDPRes[k] += Lap_noise(0, maxDSf, epsilon)

    """
    ERATO: Improved DDP
    """
    NewRes = np.zeros((numMovies, 1), float)
    for k in range(numMovies):
        maxNewDSf = 0.0
        for i in range(numDataOwners):
            maxNewDSf = max(maxNewDSf, NewDSf[k][i])
        MaxSFTable[2][k] = maxNewDSf
        NewRes[k] = TrueRes[k]
        NewRes[k] += Lap_noise(0, maxNewDSf, epsilon)

    np.savetxt("MaxSFTable", MaxSFTable, fmt='%.10f')


    """
    Compute Average Accuracy
    """
    ResAcc = np.zeros((1, 3), float)
    for k in range(numMovies):
        #DP
        ResAcc[0][0] += (1.0 - abs(DPRes[k] - TrueRes[k])/abs(DPRes[k] + TrueRes[k]))/numMovies
        #DDP
        ResAcc[0][1] += (1.0 - abs(DDPRes[k] - TrueRes[k])/abs(DDPRes[k] + TrueRes[k]))/numMovies
        #ERATO
        ResAcc[0][2] += (1.0 - abs(NewRes[k] - TrueRes[k])/abs(NewRes[k] + TrueRes[k]))/numMovies
    np.savetxt("AvgAccuracy_%d"%indexNum, ResAcc, fmt='%.10f')
    print("Accuracies [DP, DDP, ERATO]:\n", ResAcc)
