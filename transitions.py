import random
import numpy as np
from math import sin, cos, pi, exp, e, sqrt, fabs, floor
from operator import mul
from functools import reduce


__author__ = "Cesar Revelo"

def subvector(y,head,tail):
    """Construct a vector with the elements v[head], ..., v[tail-1]."""
    result = []    #result = np.zeros( len( range (0,4,1) ) )
    for i in range (int(floor(head)),int(floor(tail)),1):
    	result.append(y[i])
    return np.asarray( result )

def linear(y,A): 
    """Shift: Linear Transformation for WFG1 Transition 1."""
    return ( fabs( y-A )/fabs( floor( A-y )+A ) ) 

def b_flat(y,A,B,C): 
    """Bias: Flat Region Transformation for WFG1 Transition 2."""
    tmp1 = min( 0.0, floor( y-B ) ) * A*( B-y )/B 
    tmp2 = min( 0.0, floor( C-y ) ) * ( 1.0-A )*( y-C )/( 1.0-C )
    return ( A+tmp1-tmp2 ) 

def poly(y,alpha): 
    """Bias: Polynomial Transformation for WFG1 Transition 3."""
    return ( pow( y, alpha ) )  

def r_sum(y,w): 
    """Reduction: Weighted Sum Transformation for WFG1 Transition 4."""
    numerator = sum(y[i]*w[i] for i in range(len(y)))
    denominator = sum(w)	    
    return ( numerator / denominator ) 
