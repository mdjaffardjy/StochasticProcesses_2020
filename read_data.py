import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
#from sklearn.linear_model import LinearRegression
import SDE 

data_file = "data/brazil-TAVG-Trend.txt"

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

#important = verifier si c'est bon ou si y'a mieux, la c'est de l'interpolation lineaire donc wlh


#Trend - fitting models to the time series :

yearly_anomaly = np.array([anomaly_cpt[i-60:i+60].mean() for i in range(60,len(anomaly_cpt)-60)])
plt.plot(date,anomaly_cpt,linewidth=0.1)
plt.plot(date[60:-60],yearly_anomaly,color='red')
plt.show()
input("Press Enter to continue...") 
#Use linear regression to fit to the time series, assuming yt to be Gaussian and independently distributed. 

#constant
(fit_0, [resid, rank, sv, rcond]) = np.polynomial.polynomial.Polynomial.fit(date,anomaly_cpt,0,full=True)
plt.plot(date,fit_0(date),color='black')

#linear 
#reg = LinearRegression().fit(date, np.array(anomaly_cpt))
#predictions = reg.predict(date)

#plt.plot(date,anomaly,linewidth=0.1)
#plt.plot(date,predictions,linewidth=0.1)

(fit_1, [resid1, rank1, sv1, rcond1]) = np.polynomial.polynomial.Polynomial.fit(date,anomaly_cpt,1,full=True)
plt.plot(date,fit_1(date),color='blue')


#quadratic 
(fit_2, [resid2, rank2, sv2, rcond2]) = np.polynomial.polynomial.Polynomial.fit(date,anomaly_cpt,2,full=True)
plt.plot(date,anomaly_cpt,linewidth=0.1)
plt.plot(date,fit_2(date),color='red')
#sum of squared residuals = 467.1


#cubic polynomial 
(fit_3, [resid3, rank3, sv3, rcond3]) = np.polynomial.polynomial.Polynomial.fit(date,anomaly_cpt,3,full=True)
plt.plot(date,fit_3(date),color='green')
plt.show()
#sum of squared residuals = 464.11

#Wich regression model fits best?
#3 is not better than 2

#detrending
est = fit_2(date)
stoch = anomaly_cpt-est
yearly_stoch = np.array([stoch[i-60:i+60].mean() for i in range(60,len(stoch)-60)])
plt.plot(date, stoch, linewidth=0.1)
plt.plot(date[60:-60],yearly_stoch,color='red')
plt.show()

#power spectral density
dt = 0.1
f, S= SDE.psd(stoch,dt)
plt.plot(f,S)
plt.show()

#auto-correlation

A = SDE.autocorrel(S)
plt.plot(f,S)
plt.plot(f,A,color='red')
plt.show()

#comparison between the two = 


#Stochastic process that models the temperature anomaly















