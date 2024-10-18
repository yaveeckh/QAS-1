import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt 

def pdf(x):
    return 2 * norm.pdf(x)

def cdf(x):
    return 2 * (norm.cdf(x)-0.5)

def belly_and_tail_normal(x1, N):
    # Decide if in belly or tail
    x = np.zeros(N)

    for i in range(N):
        P = cdf(x1)
        if np.random.random() < P:
            x[i] = belly_normal(x1) # generate halfnormal rv in [0,x1] with AR method
        else:
            x[i] = tail_normal(x1) # generate halfnormal rv with fallback
    return x
    
def belly_normal(x1):
    # Choose y to be distributed uniformly in [0, x1)
    y = np.random.uniform() * x1
    u = np.random.uniform()
    
    if u <= pdf(y)/pdf(0):
        return y
    else:
        return belly_normal(x1)
    
def tail_normal(x1):
    # Sample from exponential using inversion method
    x = -np.log(np.random.uniform())/x1
    y = -np.log(np.random.uniform())

    if 2*y > x**2:
        return x+x1
    else:
        return tail_normal(x1)
    
def plot_distribution(N):
    x1 = 1
    x = belly_and_tail_normal(x1, N)
    plt.hist(x, density=True, bins=50)
    
    xn = np.linspace(0, 5, 100)
    yn = pdf(xn)
    plt.plot(xn,yn)

    plt.show()

    a = 1/(x1*pdf(0)) * cdf(x1)
    b = np.log(x1) + (x1**2/2) + 1/2* np.log(2*np.pi) + np.log(1-cdf(x1)/2)
    print(a)
    print(b)

def plot_success():

    x1 = np.linspace(0.01, 7, 1000)
    p = lambda x : 1/(x*pdf(0)) * cdf(x)
    q = lambda x: x * np.exp(x**2/2) * np.sqrt(2*np.pi) * (1-norm.cdf(x))
    En = lambda x: cdf(x) * 2 * 1/p(x) + (1-cdf(x)) * 2 * 1/q(x) + 1
    En_min = minimize(En, 1, bounds=Bounds(0, 10))
    print("Minimum amount amount of uniform rvs at x1 = {} and E[N] = {}".format(En_min.x, En_min.fun))
    
    plt.plot(x1,p(x1))
    plt.plot(x1,q(x1))
    plt.plot(x1,En(x1))
    
    plt.show()
    
if __name__ == "__main__":
    #plot_distribution(1000)
    plot_success()