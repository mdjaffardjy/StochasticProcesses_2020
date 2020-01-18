# Stochastic differential equation solver
# Euler-Maruyama method
# 5BIM Stochastic Processes 
# 2019-2020 INSA Loyn

import numpy as np
import matplotlib.pyplot as plt

''' 
Stochastic differential equation (SDE) solver using Euler-Maruyama method
For the  SDE 
    
    dx
    --  =  b(x) + sigma(x)*W(t)
    dt

where W(t) is a Gaussin white noise ( <W(t)> = 0 and <W(t)W(s)> = delta(t-s) )
The updating scheme is

    x(t+dt) = x(t) + dt*b(x(t)) + sqrt(dt)*sigma(x)*N(0,1)
'''
def solve(b,sigma,tfinal,x0,dt):
    sqrt_dt = np.sqrt(dt)
    n = int(1 + tfinal/dt)
    x = np.zeros(n)
    t = np.zeros(n)
    x[0] = x0
    for i in range(n-1):
        x[i+1] = x[i] + dt*( b(x[i]) ) + sqrt_dt*sigma(x[i])*np.random.randn()
        t[i+1] = t[i] + dt
    return x, t

def plot(t,x):
    plt.plot(t,x)
    plt.show()
    return;

'''
The Brownian motion satisfies the SDE

    dB
    --  = W(t)
    dt
'''
def b_brownian(x):
    return 0.0

def sigma_brownian(x):
    return 1.0

'''
The Ornstein-Uhlenbeck process satisfies the SDE

    dx
    -- = -theta*(x-mu) + sigma*W(t)
    dt
'''
def b_OU(x):
    return -1.0*(x - 0.5)

def sigma_OU(x):
    return 0.1

'''
The Langevin equation is the SDE

    dx
    --  b(x) + sigma(x)*W(t)
    dt
'''
def b_langevin(x):
    return x - x**3

def sigma_langevin(x):
    return 0.6

'''
Power Spectral Density and Autocorrelation of a stationary time series

The Discrete Fourier Transform (DFT) of a time series X of length N with
uniform sampling t_i = i*dt, i=0,...,N-1, can be computed with a Fast Fourier Transform

    H(f) = FFT(X)

The Power Spectral Density (PSD) of a time series is defined as

    S(f) = H(f)*H^*(f)/N

For a real time series, the Fourier transform H is symmetrical, only the first 
half of H is useful. The frequencies associated with a sampling rate 1/dt are

    f(i) = 1/dt*i/N, for i = 0,...,N-1 

The largest frequency f_N = 1/(2*dt) (i=[(N+1)/2]) is the Nyquist frequency. This is the 
largest frequency that can be correctly described in the signal. In many of our applications,
the frequencies of interest are much lower.

The autocorrelation of the time series is the inverse Fourier transform 
of the PSD 

    k(tau) = IFFT(S)
'''

def psd(x,dt):
    H = np.fft.fft(x)
    N = len(x)
    S = H*np.conjugate(H)/N
    f = 1/dt * np.arange(0,N)/N
    return f, S

def autocorrel(S):
    return np.fft.ifft(S)

if __name__ == "__main__":
    dt = 0.25
    x,t = solve(b_langevin,sigma_langevin,200.0,0.0,dt)
    N = len(x)
    '''
    keep only the first half of the frequencies:
    for real signal X, the PSD is symmetrical 
    S[n] = S[N-n], therefore
    if N is even, only S[0] to S[N/2] are needed
    if N is odd, only S[0] to S[(N-1)/2] are needed
    '''
    N2 = int(N/2)+1  
    n  = range(N2) # this goes from 0 to N/2 if even, and from 0 to (N-1)/2 if odd
    f, S = psd(x,dt)
    k = autocorrel(S)

    plt.plot(t,x)
    plt.title("x(t)")
    plt.show()

    plt.plot(f[n],np.log10(S[n]))
    # plt.plot(f,np.log10(S),'bo') # uncoment to plot the whole PSD
    plt.title("PSD")
    plt.show()

    plt.plot(t[n],k[n])
    # plt.plot(t,k,'bo') # uncomment to plot the whole PSD
    plt.show()



