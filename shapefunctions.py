import random
import numpy as np
from math import sin, cos, pi, exp, e, sqrt, fabs, floor
from operator import mul
from functools import reduce


__author__ = "Cesar Revelo"

def convex(x,m): 
    """Convex: Shape Function for WFG1."""
    result = 1.0
    M = len(x)
    result = reduce(mul, (1.0-cos(0.5*xi*pi) for xi in x[:m]), 1.0)
    if m!=1:
    	result = result*(1.0-sin(0.5*x[M-m]*pi))
    return result 

def mixed(x,A,alpha): 
    """Mixed: Shape Function for WFG1."""
    tmp = 2.0*A*pi
    return pow( 1.0-x[0]-cos( tmp*x[0] + pi/2.0 )/tmp, alpha )

        



