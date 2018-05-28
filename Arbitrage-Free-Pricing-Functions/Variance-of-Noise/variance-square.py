'''
Arbitrage-free Pricing Functions
@Part: Variance of Noise
       Simulate Arbitrage Attack
@Author: Chaoyue Niu
@Email: rvincency@gmail.com
@Reference: C. Niu, Z. Zheng, F. Wu, S. Tang, X. Gao, and G. Chen,
            "Unlocking the Value of Privacy: Trading Aggregate Statistics over Private Correlated Data", in KDD, 2018
@Settings: Variance of noise: v = 1;
           Number of diverse noises: m = 100;
           Number of total samples = 10000;
           The pricing function decreases with 1/v^2.
'''


from random import uniform
import math

def vlinear(v):
    return 1/v

def vsquare(v):
    return 1/(v**2)

def vsqrt(v):
    return 1/(math.sqrt(v))


if __name__=="__main__":
    v = 1
    m = 100
    testtimes = 10000
    #percentage 
    res = [0.0] * 10000 
    for t in range(0, testtimes):
        arr = []
        sum0 = 0.0
        for i in range(0,m):
            tmp = uniform(v, (m ** 2) * v - (m - 1) * v)
            sum0 += tmp
            arr.append(tmp)
        for i in range(0,m):
            arr[i] = (arr[i] / sum0) * (m**2 * v)
        #if \pi -> 1/v^2
        pay = vsquare(v)
        pay_sum = 0.0
        for i in range(0,m):
            pay_sum += vsquare(arr[i])
        #ratio between attack cost and original price
        print(pay_sum/pay)
        #record sample percentage
        res[int(pay_sum/pay)] += 1.0/testtimes * 100
    print(res)
    file = open("res-square", 'w')
    for item in res:
        file.write("%s\n" % item)
    file.close()
