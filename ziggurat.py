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
    P = cdf(x1)
    if np.random.random() < P:
        return belly_normal(x1) # generate halfnormal rv in [0,x1] with AR method
    else:
        return tail_normal(x1, table, N) # generate halfnormal rv with fallback
    
def belly_normal(x1):
    # Choose y to be distributed uniformly in [0, x1)
    y = np.random.uniform() * x1
    u = np.random.uniform()
    
    if u <= pdf(y)/pdf(0):
        return y
    else:
        return belly_normal(x1)
    
def tail_normal(x1, table, N):

    xt, yt, _ = table
    
    i = np.random.randint(N)
    x = np.random.uniform() * (xt[i-1] - x1) + x1
    
    if x < xt[i]:
        return x
    elif i == 0:
        return -1
    else:
        y = np.random.uniform() * (yt[i] - yt[i-1]) + yt[i-1]
        
        if y < pdf(x):
            return x
        else:
            return tail_normal(x1, table, N)

def try_xy(x1, r, N, tolerance):
    x = np.zeros(N)
    y = np.zeros(N)
    A = np.zeros(N)

    # Set rightmost points
    x[0] = x1 + r
    y[0] = pdf(x[0])
    A[0] = (x[0]-x1) * y[0] + Q(x[0])

    status = 0
    for i in range(1, N):
        y[i] = y[i-1] + A[0]/(x[i-1] - x1)

        # Indicates that r is too small
        if y[i] > pdf(x1):
            status = 1
            break

        x[i] = inv_pdf(y[i])
        A[i] = (x[i-1] - x1) * (y[i]-y[i-1])
    
    # status = 1: r too small, 2: r too big, 0: r falls within tollerance
    if np.abs(y[-1] - pdf(x1)) < tolerance:
        status = 0
    elif y[-1] > pdf(x1) or status == 1:
        status = 1
    else:
        status = 2

    return status, x, y, A

def ziggurat_table(x1, N):
    a = 0.01
    b = 10
    tolerance = 1e-5

    x, y, A = 0, 0, 0
    while True:
        c = (a + b)/2
        status, x, y, A = try_xy(x1, c, N, tolerance)

        if status == 1:
            a = c
        if status == 2:
            b = c
        if status == 0:
            return x, y, A

def plot_boxes():
    x,y,A = ziggurat_table(1, 17)
    plt.plot(x,y, 'bo')

    xt = np.linspace(0, 5, 10000)
    yt = pdf(xt)
    plt.plot(xt,yt)
    plt.show()

def plot_distribution(N):
    x = np.zeros(N)
    x1 = 1

    S = 256
    table = ziggurat_table(x1, S)
    
    for i in range(N):
        x[i] = belly_and_tail_normal(x1, table, S)
        
    print(x)
    plt.hist(x, density=True, bins=1000)
    
    xn = np.linspace(0, 5, 100)
    yn = pdf(xn)
    plt.plot(xn,yn)

    plt.show()

    
    
if __name__ == "__main__":
    plot_distribution(1000000)
    #plot_boxes()