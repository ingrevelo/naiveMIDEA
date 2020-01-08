"""
Regroup typical EC benchmarks functions to import easily and benchmark
examples.

Python Implementation of benchmarks functions to import easily from Ipython Notebook.
These implementation from .java problems are part of the research project at the Advanced 
Topics in Evolutionary Computation: Theory and Practice course. Using this problems we are
going to test the performance (using Convergence to Pareto True, Diversity and Hypervolume 
as indicators) of NSGA-II, SPEA-II and likely the Naive MIDEA MOEAs.
"""
import random
import copy
import transitions
import shapefunctions
import math
import numpy as np
from math import sin, cos, pi, exp, e, sqrt
from operator import mul
from functools import reduce

def problem1(individual):
    """
    Implements the first test problem ("bi-objetive Sphere Model").
    Variables = 2
    Objetives = 2  
  
    f(x) = (x'x, (x - a)'(x - a))' with a = (0, 1)' and a, x element of R^2.

    Bounds [0, 1]
    @author Cesar Revelo
    """
    a  = [0,1]
    f1 = (individual[0] ** 2) + (individual[1] ** 2)
    f2 = ((individual[0] - a[0]) ** 2) + ((individual[1] - a[1]) ** 2)
    return f1, f2

def problem2(individual, obj):
    """ 
    Implements the second test problem (this is more or less just a copy of jmetal.problems.DTLZ.DTLZ3).
    Variables = 10
    Objetives = 2 

    It returns a tuple of *obj* values. 
    The individual must have at least *obj* elements.
    From: K. Deb, L. Thiele, M. Laumanns and E. Zitzler. Scalable Multi-Objective 
    Optimization Test Problems. CEC 2002, p. 825 - 830, IEEE Press, 2002.
    
    g(\\mathbf{x}_m) = 100\\left(|\\mathbf{x}_m| + \sum_{x_i \in \\mathbf{x}_m}\\left((x_i - 0.5)^2 - \\cos(20\pi(x_i - 0.5))\\right)\\right)`
    
    f_{\\text{DTLZ3}1}(\\mathbf{x}) = (1 + g(\\mathbf{x}_m)) \\prod_{i=1}^{m-1} \\cos(0.5x_i\pi)`
    
    f_{\\text{DTLZ3}2}(\\mathbf{x}) = (1 + g(\\mathbf{x}_m)) \\sin(0.5x_{m-1}\pi ) \\prod_{i=1}^{m-2} \\cos(0.5x_i\pi)`
    
    f_{\\text{DTLZ3}m}(\\mathbf{x}) = (1 + g(\\mathbf{x}_m)) \\sin(0.5x_{1}\pi )`
    
    Where :math:`m` is the number of objectives and :math:`\\mathbf{x}_m` is a
    vector of the remaining attributes :math:`[x_m~\\ldots~x_n]` of the
    individual in :math:`n > m` dimensions.  
  
    Bounds [0, 1].
    @author Cesar Revelo
    """
    xc = individual[:obj-1]
    xm = individual[obj-1:]
    g = 100 * (len(xm) + sum((xi-0.5)**2 - cos(20*pi*(xi-0.5)) for xi in xm))
    f = [(1.0+g) *  reduce(mul, (cos(0.5*xi*pi) for xi in xc), 1.0)]
    f.extend((1.0+g) * reduce(mul, (cos(0.5*xi*pi) for xi in xc[:m]), 1) * sin(0.5*xc[m]*pi) for m in range(obj-2, -1, -1))
    return f

def problem3(individual):
    """ 
    Implements the third test problem ("DENT"), at lambda = 0.85
    Variables = 2
    Objetives = 2
    
    d(x)  = lambda * exp(-(x1 - x2)**2)
    f1(x) = 1/2 * (sqrt(1 + (x1 + x2)**2) + sqrt(1 + (x1 - x2)**2) + x1 - x2 ) + d
    f2(x) = 1/2 * (sqrt(1 + (x1 + x2)**2) + sqrt(1 + (x1 - x2)**2) - x1 + x2 ) + d
    
    Bounds [-1.5, 1.5]
    @author Cesar Revelo
    """
    lbda  = 0.85
    d  = lbda * exp(-(individual[0] - individual[1]) ** 2)  
    f1 = 0.5 * (sqrt(1 + (individual[0] + individual[1]) ** 2) + sqrt(1 + (individual[0] - individual[1]) ** 2) + individual[0] - individual[1]) + d
    f2 = 0.5 * (sqrt(1 + (individual[0] + individual[1]) ** 2) + sqrt(1 + (individual[0] - individual[1]) ** 2) - individual[0] + individual[1]) + d
    return f1, f2

def problem4(individual):
    """ 
    Implements the fourth test problem (this is more or less just a copy of jmetal.problems.ZDT.ZDT3).
    Variables = 20
    Objetives = 2
    
    g(\\mathbf{x}) = 1 + \\frac{9}{n-1}\\sum_{i=2}^n x_i`
    f_{\\text{ZDT3}1}(\\mathbf{x}) = x_1`
    f_{\\text{ZDT3}2}(\\mathbf{x}) = g(\\mathbf{x})\\left[1 - \\sqrt{\\frac{x_1}{g(\\mathbf{x})}} - \\frac{x_1}{g(\\mathbf{x})}\\sin(10\\pi x_1)\\right]`

    Bounds [0, 1].
    @author Cesar Revelo
    """
    g  = 1.0 + 9.0*sum(individual[1:])/(len(individual)-1)
    f1 = individual[0]
    f2 = g * (1 - sqrt(f1/g) - f1/g * sin(10*pi*f1))
    return f1, f2

def WFG1(Z, k, l, M):
    """ 
    Implements the fifth test problem (WFG.WFG1).
    2 position-related parameters (k)
    4 distance-related parameters (l)
    2 objectives (M)
    6 Variables : k+l

    * Reference: Simon Huband, Luigi Barone, Lyndon While
    * A Review of Multiobjetive Test Problems and a Scalable Test Problem Toolkit: 
    * IEEE TRANSACTIONS ON EVOLUTIONARY COMPUTATION, 
    * Volume 10, No. 5, October 2006
    
    Bounds [0, 1].
    @author Cesar Revelo
    """
    y = np.zeros( len(Z) )
    S = np.zeros( M )
    D = 1.0
    A = np.zeros( M-1 )
    
    #Scaling Constants for each Objetive
    for i in range( M ):
        S[i] = 2.0 * ( i+1 )
           
    #Degeneracy Constants  
    for i in range( M-1 ):
        A[i] = 1.0

    #Normalize Indvidual Z (Zi=Zi/2i) 
    for i in range( len(Z) ):
        y[i] = Z[i]/( 2.0*(i+1) )            
        
    #Transition y-->t1: linear(y(k+1,n),0.35)
    t1 = copy.copy( y )
    for i in range(k, len(y), 1):
        t1[i] = transitions.linear(y[i], 0.35)
    
    #Transition t1-->t2: b_flat(y(k+1,n),0.8,0.75,0.85)
    t2 = copy.copy( t1 )
    for i in range(k, len(t2), 1):
        t2[i] = transitions.b_flat(y[i], 0.8, 0.75, 0.85)
    
    #Transition t2-->t3: poly(y(1,n),0.8,0.75,0.85)
    t3 = copy.copy( t2 )
    for i in range( len(t3) ):
        t3[i] = transitions.poly(y[i], 0.02)
    
    #Transition t3-->t4: r_sum({y(i-1)k/(M-1)+1,..,y(ik)/(M-1)}, r_sum({y(k+1),..,y(n)},{2(k+1),..,2n})
    t4 = copy.copy( t3 )
    w  = np.zeros( len(t4) ) 
    t  = []
    for i in range( len(t4) ):
        w[i] = 2 * ( i+1 )

    for i in range( M-1 ):
        head  = i*k/( M-1 )
        tail  = ( i+1 )*k/( M-1 )
        y_sub = transitions.subvector(t4, head, tail)
        w_sub = transitions.subvector(w, head, tail)
        t.append(transitions.r_sum(y_sub, w_sub))
    
    y_sub = transitions.subvector(t4, k, len(t4))
    w_sub = transitions.subvector(w, k, len(t4))
    t.append(transitions.r_sum(y_sub, w_sub))
    t = np.asarray( t )
    
    #Computation X Vector X-->(t,A)
    x = []
    for i in range( M-1 ):
        tmp1 = max(t[-1], A[i])
        x.append(tmp1*( t[i]-0.5 ) + 0.5)
    x.append(t[-1])
    x = np.asarray( x )
    
    #Computation shape function for each X element
    h = []
    for m in range(1, M, 1):
        h.append(shapefunctions.convex(x, m))
    h.append(shapefunctions.mixed(x, 5, 1))
    h = np.asarray( h )  
    
    #Computation the scaled fitness values for a WFG1
    result = []
    for i in range( M ):
        result.append(D*x[-1] + S[i]*h[i])
    return result    
