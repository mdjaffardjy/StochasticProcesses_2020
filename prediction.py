#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 16:27:11 2020

@author: tmorealdeb
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import math


import SDE 
#Loading data
data_file = "data/47.42N-10.66E-TAVG-Trend.txt"

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

#fill out missing data by interpolation
anomaly_cpt = pd.Series(anomaly)
anomaly_cpt = anomaly_cpt.interpolate()
#sum of squared errors (Augsburg) : 10151.73


(fit_2, [resid2, rank2, sv2, rcond2]) = np.polynomial.polynomial.Polynomial.fit(date,anomaly_cpt,2,full=True)

#detrending
est = fit_2(date)
stoch = anomaly_cpt-est

dt=1/12
date_pred=[date[-1]]

for i in range(1096) :
    date_pred=np.append(date_pred, date_pred[-1]+dt)


#Values obtained by simulated annealing
theta=9.835211430103028
mu=0.
sigma_ou=5.929760525958848
xt_pred=fit_2(date_pred)

temp_2100=[]
for i in range(10000) :
    #Simulating data
    yt3=[stoch[stoch.last_valid_index()]]
    
    for t in date_pred[1:] :
        yt3.append(yt3[-1]-dt*theta*(yt3[-1]-mu)+math.sqrt(dt)*sigma_ou*np.random.normal( ))
    
    sim3 = xt_pred + yt3
    #yt3 = np.array(yt3)
    
    ten_year_sim = np.array([sim3[i-60:i+60].mean() for i in range(60,len(sim3)-60)])
   # ten_year_avg = np.array([anomaly_cpt[i-60:i+60].mean() for i in range(60,len(anomaly_cpt)-60)])
    temp_2100.append(ten_year_sim[date_pred[60:-60]==date_pred[60:-60][-1]])

temp_2100=np.array(temp_2100)
proba = len(temp_2100[temp_2100>2.5])/len(temp_2100)
print(proba)

#plt.plot(date, anomaly_cpt, linewidth=.1, color='red')
#plt.plot(date[60:-60], ten_year_avg, color='red')

#plt.plot(date_pred[np.where(date_pred==date[-1])[0][0]:], sim3[np.where(date_pred==date[-1])[0][0]:], color='green', linewidth=.1)
#plt.plot(date_pred[np.where(date_pred==date[-1])[0][0]:-60], ten_year_sim[np.where(date_pred==date[-61])[0][0]:], color='green')