import scipy.stats as stats
import numpy as np

# Author: Andrea Urgolo <andrea.urgolo@uniud.it>

def jaccard_test(x,y):    
    """ 
        Jaccard/Tanimoto similarity test as defined in 
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6929325
        
        :param x: list of boolean values
        :param y: list of boolean values
        
        :return: (Jaccard/Tanimoto coefficient, p_value)
    """
    
    m=len(x)
    
    if m != len(y):
        raise ValueError("Array of different size: " + str(m) + ",  " + str(len(y)))
        
    if m == 0:
        raise ValueError("0-sized vectors not supported")
    
    
    # fix coldstart problems
    x += [1,0]
    y += [0,0]
    m+=2
    
    # extimate n1, n2, n3, n4    
    n1 = np.sum(np.logical_and(x,y))
    n2 = np.sum(np.logical_and(x,np.subtract(1,y)))
    n3 = np.sum(np.logical_and(np.subtract(1,x), y))
    n4 = m-n1-n2-n3
    
    # n5=m-n4 # np.sum(nplogical_or(x, y))    
    
    # computing Jaccard/Tanimoto coef. T
    if (n1+n2+n3) > 0:
        T = n1/(n1+n2+n3)
    else: 
        raise ValueError("No positive values in input arrays") 
        
        
        
    # computing p_val
    px = np.sum(x) / m
    py = np.sum(y) / m
    epsilon = 1e-10 # for numerical stability
    q1 = px*py # the probability that both x and y have ones
    q2 = px+py-(2*px*py) # the probability that only one of two vectors has one.
    sigma = np.sqrt((q1*q2*(1-q2))/((q1+q2+epsilon)**3))
    p_val = 2 * stats.norm.cdf((np.sqrt(m)/(sigma+epsilon)) * (T - q1/(q1+q2+epsilon))) - 1
    
    return T, p_val