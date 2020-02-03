import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
#from sklearn.linear_model import LinearRegression
from scipy import optimize
import math
import annealing

def test_func(x, a, b):
    return a * np.sin(b * x)


import SDE 
#Loading data
data_file = "data/47.42N-10.66E-TAVG-Trend.txt"
#data_file = "data/brazil-TAVG-Trend.txt"

# initialize lists

year = []
month = []
anomaly  = []
# import data from txt file
with open(data_file,'r', encoding ='latin1') as f:
  reader = csv.reader(f)
  for line in reader:
    lin = line[0].split()
    if len(lin) == 12 and lin[0] != "%": # skip non-data lines
      year.append(float(lin[0]))
      month.append(float(lin[1]))
      anomaly.append(float(lin[2]))
      
# convert to numpy array
year = np.array(year)
month = np.array(month)
anomaly = np.array(anomaly)
date = year + (month-0.5)/12.0

plt.plot(date,anomaly,linewidth=0.1)
plt.show()

#fill out missing data by interpolation
anomaly_cpt = pd.Series(anomaly)
anomaly_cpt = anomaly_cpt.interpolate()
#sum of squared errors (Augsburg) : 10151.73


#Trend - fitting models to the time series :

yearly_anomaly = np.array([anomaly_cpt[i-60:i+60].mean() for i in range(60,len(anomaly_cpt)-60)])
plt.plot(date,anomaly_cpt,linewidth=0.1)
plt.plot(date[60:-60],yearly_anomaly,color='red')
plt.show()
#input("Press Enter to continue...") 
#Use linear regression to fit to the time series, assuming yt to be Gaussian and independently distributed. 

#constant
(fit_0, [resid, rank, sv, rcond]) = np.polynomial.polynomial.Polynomial.fit(date,anomaly_cpt,0,full=True)
plt.plot(date,fit_0(date),color='black')

#linear
(fit_1, [resid1, rank1, sv1, rcond1]) = np.polynomial.polynomial.Polynomial.fit(date,anomaly_cpt,1,full=True)
plt.plot(date,fit_1(date),color='blue')
#sum of squared residuals (Augsburg)= 9965.16

#quadratic 
(fit_2, [resid2, rank2, sv2, rcond2]) = np.polynomial.polynomial.Polynomial.fit(date,anomaly_cpt,2,full=True)
plt.plot(date,anomaly_cpt,linewidth=0.1)
plt.plot(date,fit_2(date),color='red')
#sum of squared residuals (Brazil)= 467.1
#sum of squared residuals (Augsburg)= 9741.23

#cubic polynomial 
(fit_3, [resid3, rank3, sv3, rcond3]) = np.polynomial.polynomial.Polynomial.fit(date,anomaly_cpt,3,full=True)
plt.plot(date,fit_3(date),color='green')
plt.show()
#sum of squared residuals (Brazil) = 464.11
#sum of squared residuals (Augsburg)= 9715.18

#Wich regression model fits best?
#3 is not better than 2 so we choose degree 22

#detrending
est = fit_2(date)
stoch = anomaly_cpt-est
yearly_stoch = np.array([stoch[i-60:i+60].mean() for i in range(60,len(stoch)-60)])
plt.plot(date, stoch, linewidth=0.1)
plt.plot(date[60:-60],yearly_stoch,color='red')
#detrended mean (Brazil) : 1.48e-16
#detrended mean (Augsburg) : 1.38e-16

#params, params_covariance = optimize.curve_fit(test_func, x_data, y_data, p0=[2, 2])
plt.show()
input()


#power spectral density
dt = 1/12
f, S= SDE.psd(stoch,dt)
plt.plot(f,S)
plt.show()
#The PSD shows a peak at frequency 1, suggesting the presence of an annual cyclic variation in the data


#auto-correlation

A = SDE.autocorrel(S)
#plt.plot(f,S)
plt.plot(f,A,color='red')
plt.show()

#comparison between the two = 

#Stochastic process that models the temperature anomaly
xt=fit_2(date)

input("Start annealing")
#Stochastic part will be simulated using Ornstein-Uhlenbeck process
#The parameter mu is set at 0 (mean)
#The parameters theta and sigma are optimized using simulated annealing in such way that the difference between
#variance of detrended data and variance of stochastic process (averaged over 10 realizations) is minimal.
def cost_function(x) :
    variances=[]
    for j in range(10) :
        yt3=[0]
        theta=x[0]
        mu=0.
        sigma_ou=x[1]
    
        dt=1/12
        for t in date[1:] :
            yt3.append(yt3[-1]-dt*theta*(yt3[-1]-mu)+math.sqrt(dt)*sigma_ou*np.random.normal( ))
        
        yt3 = np.array(yt3)
        #f_sim, S_sim= SDE.psd(yt3,dt)
        variances.append(yt3.var())
    variances=np.array(variances)
    return abs(stoch.var()-variances.mean())
    #return sum((np.real(S)-np.real(S_sim))**2)
    #return abs(f[S==max(S)]-f_sim[S_sim==max(S_sim)])[0]

state, c, states, costs = annealing.annealing([5., 5.],
              cost_function,
              annealing.random_neighbour,
              annealing.acceptance_probability,
              annealing.temperature,
              maxsteps=10000,
              debug=False)

print("theta : "+str(state[0]))
print("sigma : "+str(state[1]))

#ornstein-uhlenbeck
#Values for theta and mu are set after optimization by simulated annealing (see above)
yt3=[stoch[0]]
theta=state[0]
mu=0.
sigma_ou=state[1]

dt=1/12
for t in date[1:] :
    yt3.append(yt3[-1]-dt*theta*(yt3[-1]-mu)+math.sqrt(dt)*sigma_ou*np.random.normal( ))

sim3 = xt + yt3
yt3 = np.array(yt3)


yearly_sim = np.array([sim3[i-60:i+60].mean() for i in range(60,len(sim3)-60)])

plt.plot(date, anomaly_cpt, linewidth=0.1, color='red')
plt.plot(date, sim3, linewidth=0.1, color='green')

plt.plot(date[60:-60], yearly_anomaly, color='red')
plt.plot(date[60:-60], yearly_sim, color='green')
plt.show()
print("yt3 : "+str(yt3.var()))
print("stoch : "+str(stoch.var()))
input()

dt = 1/12
f_sim, S_sim= SDE.psd(yt3,dt)
A_sim = SDE.autocorrel(S_sim)
plt.plot(f_sim,S_sim)
plt.show()

