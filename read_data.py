import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from sklearn.linear_model import LinearRegression
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

yearly_anomaly = np.array([anomaly_cpt[i-120:i+120].mean() for i in range(120,len(anomaly_cpt)-120)])
plt.plot(date,anomaly_cpt,linewidth=0.1)
plt.plot(date[120:-120],yearly_anomaly,color='red')
plt.show()

#Use linear regression to fit to the time series, assuming yt to be Gaussian and independently distributed. 

#constant


#linear 
reg = LinearRegression().fit(date, np.array(anomaly_cpt))
predictions = reg.predict(date)

plt.plot(date,anomaly,linewidth=0.1)
plt.plot(date,predictions,linewidth=0.1)
plt.show()

#quadratic 
#cubic polynomial 

#Wich regression model fits best?


#de-trending the time data

#power spectral density
dt = 0.1
#f, S= SDE.psd(detrended,dt)
#plt.plot(f,S)
#plt.show()

#auto-correlation

#A = autocorrel(S)
#plt.plot(f,S)
#plt.plot(f,A)
#plt.show()

#comparison between the two = 


#Stochastic process that models the temperature anomaly















