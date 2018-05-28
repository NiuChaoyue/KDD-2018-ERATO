'''
Arbitrage-free Pricing Functions
@Part: Weight Vector
       Linear Regression
@Author: Chaoyue Niu
@Email: rvincency@gmail.com
@Reference: C. Niu, Z. Zheng, F. Wu, S. Tang, X. Gao, and G. Chen,
            "Unlocking the Value of Privacy: Trading Aggregate Statistics over Private Correlated Data", in KDD, 2018
@Settings: numDataOwners controls the dimension of weight vector
'''


import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


if __name__=="__main__":
    """
    Load ratings.dat
    """
    f1 = open("ratings.dat")
    ratings = np.zeros((4000, 6100), int)
    while True:
        line = f1.readline()
        if not line:
            break
        linetmp = line.split("::")
        userID = int(linetmp[0])
        moiveID = int(linetmp[1])
        moiveScore = int(linetmp[2])
        ratings[moiveID][userID] = moiveScore
    f1.close()

    """
    Load Target Variables
    """
    f2 = open("targetVariables")
    targetVar = np.zeros((4000,1), float)
    for i in range(4000):
        line = f2.readline()
        if not line:
            break
        targetVar[i] = float(line[0])
    print(targetVar)

    """
    Slicing part of data for training
    """
    numDataOwners = 6000
    numTest = 10
    weightVecs = np.zeros((numDataOwners, numTest), float)
    index = 0
    for t in range(numTest):
        numMovies = 30
        newX = ratings[(1 + t * numMovies):(numMovies + 1 + t * numMovies), 1: (numDataOwners + 1)]
        newY = targetVar[(1 + t * numMovies):(numMovies + 1 + t * numMovies)]

        X_train, X_test, y_train, y_test = train_test_split(newX, newY, test_size = 0.2, random_state = 24)

        regr = linear_model.LinearRegression()
        regr.fit(X_train, y_train)

        """
        Predictations
        """
        y_pred = regr.predict(X_test)
        print("y_test: \n", y_test)
        print("y_pred: \n", y_pred)
        mse = mean_squared_error(y_test, y_pred)
        print("Mean squared error: %.2f" % mse)

        regrCoef = regr.coef_
        if(mse < 100):
            for i in range(numDataOwners):
                weightVecs[i][index] = regrCoef[0][i]
            index += 1
    np.savetxt("weightVecs_%d" % numDataOwners, weightVecs, fmt='%.10f')