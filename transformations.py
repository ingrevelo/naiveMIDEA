__author__ = "Cesar David Revelo"

from math import *


def b_poly(y, alpha=0.02):
    """Bias: Polynomial Transformation."""
    return pow(y, alpha)


def b_flat(y, A, B, C):
    """Bias: Flat Region Transformation."""
    tmp1 = min(0.0, floor(y - B)) * A * (B - y) / B
    tmp2 = min(0.0, floor(C - y)) * (1.0 - A) * (y - C) / (1.0 - C)
    return A + tmp1 - tmp2


def p_dependent(y, y_deg, A=0.98/49.98, B=0.02, C=50.0):
    """Bias: Parameter Dependent Transformation."""
    aux = A - (1.0-2.0*y_deg)*fabs(floor(0.5-y_deg)+A)
    return pow(y, B+(C-B)*aux)


def s_linear(y, A):
    """Shift: Linear Transformation."""
    return fabs(y-A) / fabs(floor(A-y) + A)


def s_deceptive(y, A=0.35, B=0.001, C=0.05):
    """Shift: Parameter Deceptive Transformation."""
    tmp1 = floor(y-A+B)*(1.0-C+(A-B)/B)/(A-B)
    tmp2 = floor(A+B-y)*(1.0-C+(1.0-A-B)/B)/(1.0-A-B)
    return 1.0 + (fabs(y-A)-B)*(tmp1+tmp2+1.0/B)


def s_multi_modal(y, A, B, C):
    """Shift: Parameter Multi-Modal Transformation."""
    tmp1 = fabs(y-C)/( 2.0*(floor(C-y)+C))
    tmp2 = (4.0*A+2.0)*pi*(0.5-tmp1)
    return (1.0 + cos(tmp2) + 4.0*B*pow(tmp1, 2.0)) / (B+2.0)


def r_sum(y, w):
    """Reduction: Weighted Sum Transformation."""
    numerator = sum(y[i] * w[i] for i in range(len(y)))
    denominator = sum(w)
    return numerator/denominator


def r_non_sep(y, A):
    """Reduction: Non-Separable Transformation."""
    aux = len(y)
    numerator = 0.0
    for j in range(aux):
        numerator += y[j]
        for k in range(A-1):
            numerator += fabs(y[j]-y[(1+j+k) % aux])
    tmp = ceil(A/2.0)
    denominator = aux*tmp*(1.0 + 2.0*A - 2*tmp)/A
    return numerator/denominator


def subvector(y, head, tail):
    """Construct a vector with the elements v[head], ..., v[tail-1]."""
    result = []
    for i in range(head, tail, 1):
        result.append(y[i])
    return result