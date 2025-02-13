from math import ceil
from scipy.stats import poisson

"""
This script computes properties of an Erdos-Renyi random graph. We calculate
the average degree, the variance, and the number of expected nodes with degree
2 x the average degree
"""
def erdos_renyi_properties(N, p):
    # compute the average degree
    avgDeg = (N - 1) * p
    
    # compute the variance
    variance = (N - 1) * p * (1 - p)  # Binomial variance formula
    
    # compute expected number of nodes with degree at least 2 times avgDeg
    twoTimesNodes = ceil(2 * avgDeg)  # The degree threshold

    # compute using Poisson approximation
    probabilityOfK = 1 - sum(poisson.pmf(k, avgDeg) for k in range(twoTimesNodes))
    
    # probability of having at least k times N
    expectedNodes = N * probabilityOfK 
    
    return avgDeg, variance, expectedNodes

N = 4000  # Number of nodes
p = 0.001  # Edge probability

a, b, c = erdos_renyi_properties(N, p)
print(f"(a) Average degree: {a}")
print(f"(b) Variance of degree: {b}")
print(f"(c) Expected nodes with degree >= 2 * avg degree: {c}")
