import numpy as np
import matplotlib.pyplot as plt
import csv

#importing and cleaning the data

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

plt.plot(date,anomaly,linewidth=0.1)
plt.show()

#fill out missing data by interpolation

#no missing data...

#Trend - fitting models to the time series :

#Use linear regression to fit to the time series, assuming yt to be Gaussian and independently distributed. 

#constant
#linear
#quadratic 
#cubic polynomial 

#Wich regression model fits best?






















