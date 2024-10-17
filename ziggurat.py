import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt 

def pdf(x):
    return 2*norm.pdf(x)

def inv_pdf(y):
    return np.sqrt(-2*np.log(np.sqrt(np.pi/2)*y))

def cdf(x):
    return 2*(norm.cdf(x)-0.5)

def Q(x):
    return 1-cdf(x)

def belly_and_tail_normal(x1, N):
    # Decide if in belly or tail
    for i in range(N):
        P = cdf(x1)
        if np.random.random() < P:
            return belly_normal(x1) # generate halfnormal rv in [0,x1] with AR method
        else:
            return -1
            return tail_normal(x1, table, S) # generate halfnormal rv with fallback

    
def belly_normal(x1):
    """Generate halfnormal sample via acceptance-rejection sampling.

    Args:
        x1 (float): X values lower than this threshold get generated using acceptance-rejection sampling, after this threshold Ziggurat with fallback is used.        

    Returns:
        float: Halfnormal sample.
    """
    # Choose y to be distributed uniformly in [0, x1)
    y = np.random.uniform() * x1
    u = np.random.uniform()
    
    if u <= pdf(y)/pdf(0):
        return y
    else:
        return belly_normal(x1)
    
def tail_normal(x1):

    

if __name__ == "__main__":
    plot_distribution(50000)
    #plot_boxes()