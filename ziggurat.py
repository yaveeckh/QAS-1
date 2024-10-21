import numpy as np
from scipy.stats import norm, chi2
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt 

def pdf(x):
    return 2 * norm.pdf(x)

def cdf(x):
    return 2 * (norm.cdf(x)-0.5)

def belly_and_tail_normal(x1, N):
    x = np.zeros(N)

    for i in range(N):
        # Check if in belly or tail
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
    
def c2test(N, k, alpha):
    x = belly_and_tail_normal(1, N)
    
    # Cap range such that there can be no division by zero in test
    Nj, edges = np.histogram(x, bins=k, range=[0, 8])
    expected = np.zeros(k)

    # Recalculate N to not include outliers
    N = np.sum(Nj)

    # Calculate expected values in each interval
    for i in range(k):
        expected[i] = (cdf(edges[i+1]) - cdf(edges[i])) * N

    # Calculate Chi^2 value
    X2 = sum((xj-expected[j])**2/(expected[j]) for (j,xj) in enumerate(Nj))

    # Test
    print(chi2.ppf(alpha, k-1))
    if X2 < chi2.ppf(alpha, k-1):
        print("Passed")
    else:
        print("Failed")

    
def plot_distribution(N):
    x1 = 1
    x = belly_and_tail_normal(x1, N)
    plt.hist(x, density=True, bins=50)
    
    xn = np.linspace(0, 5, 100)
    yn = pdf(xn)
    plt.plot(xn,yn)

    plt.show()

def plot_success():

    x1 = np.linspace(0.1, 7, 1000)
    p = lambda x : 1/(x*pdf(0)) * cdf(x)
    q = lambda x: x * np.exp(x**2/2) * np.sqrt(2*np.pi) * (1-norm.cdf(x))
    En = lambda x: cdf(x) * 2 * 1/p(x) + (1-cdf(x)) * 2 * 1/q(x) + 1
    En_min = minimize(En, 1, bounds=Bounds(0, 10))
    print("Minimum amount amount of uniform rvs at x1 = {} and E[N] = {}".format(En_min.x, En_min.fun))
    
    plt.plot(x1,p(x1), label="$p(x_1)$")
    plt.plot(x1,q(x1), label="$q(x_1)$")
    plt.xlabel("$x_1$")
    plt.legend()
    plt.title("Succes probabilities of one trial in belly and tail")
    plt.show()

    plt.plot(x1,En(x1), label="$E_{x_1}[N]$")
    plt.xlabel("$x_1$")
    plt.legend()
    plt.title("Mean number of required uniform samples per half-normal sample")
    plt.show()


    
    
if __name__ == "__main__":
    #plot_distribution(100000)
    #plot_success()
    c2test(5, 10, 0.95)