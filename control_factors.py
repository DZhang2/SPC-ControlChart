import pandas as pd
from math import sqrt

factors = pd.read_csv("control_factors.csv")

def getc4(n):
    return factors["c4_list"][n-2]

def getd2(n):
    return factors["d2_list"][n-2]
    
def getd3(n):
    return factors["d3_list"][n-2]

def getD1(n, L):
    return max(0, getd2(n) - L * getd3(n))
    
def getD2(n, L):
    return getd2(n) + L * getd3(n)

def getD3(n, L):
    return max(0, getD1(n, L)/getd2(n))

def getD4(n, L):
    return getD2(n, L)/getd2(n)

def getB3(n, L):
    c4 = getc4(n)
    return max(0, 1 - L * sqrt(1-c4**2) / c4)

def getB4(n, L):
    c4 = getc4(n)
    return 1 + L * sqrt(1-c4**2) / c4

def getB5(n, L):
    c4 = getc4(n)
    return max(0, c4 - L * sqrt(1-c4**2))

def getB6(n, L):
    c4 = getc4(n)
    return c4 + L * sqrt(1-c4**2)

def getA(n, L):
    return L/sqrt(n)

def getA2(n, L):
    return getA(n, L)/getd2(n)

def getA3(n, L):
    return L/(getc4(n) * sqrt(n))

