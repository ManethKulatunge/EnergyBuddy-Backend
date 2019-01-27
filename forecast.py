import pandas as pd
import os, math
import numpy as np 
import matplotlib.pyplot as plt 
import sklearn
from sklearn.cluster import KMeans

file_read = pd.read_csv('weather_description.csv')
file_read = file_read.drop(columns=['datetime', 'Beersheba', 'Tel Aviv District', 'Eilat', 'Haifa','Nahariyya', 'Jerusalem'])
print(file_read.columns, file_read.shape)
print(file_read.head(5))
 
file_read = file_read.fillna(0)
cities = ['Vancouver', 'Portland', 'San Francisco', 'Seattle',
       'Los Angeles', 'San Diego', 'Las Vegas', 'Phoenix', 'Albuquerque',
       'Denver', 'San Antonio', 'Dallas', 'Houston', 'Kansas City',
       'Minneapolis', 'Saint Louis', 'Chicago', 'Nashville', 'Indianapolis',
       'Atlanta', 'Detroit', 'Jacksonville', 'Charlotte', 'Miami',
       'Pittsburgh', 'Toronto', 'Philadelphia', 'New York', 'Montreal',
       'Boston']

# dictionary containing reindexing for data
solar_desc_index = {'mist':3, 'sky is clear':10, 'few clouds':8, 'overcast clouds':5,
 'scattered clouds':6, 'broken clouds':5, 'light intensity drizzle':0, 'light rain':0,
 'fog':4, 'haze':5, 'heavy snow':0, 'dust':7, 'proximity thunderstorm':0,
 'thunderstorm with rain':0, 'thunderstorm':0, 'thunderstorm with heavy rain':0,
 'heavy intensity rain':0, 'moderate rain':0, 'drizzle':0,
 'heavy intensity drizzle':0, 'thunderstorm with light rain':0,
 'proximity thunderstorm with rain':0, 'thunderstorm with heavy drizzle':0,
 'very heavy rain':0, 'smoke':5, 'light snow':0, 'snow':0, 'squalls':8,
 'thunderstorm with light drizzle':0, 'tornado':3, 'thunderstorm with drizzle':0,
 'proximity shower rain':0}

file_read = file_read.replace(solar_desc_index)
file_read = file_read.mean(axis=0)
print("file_read", file_read)

file2_read = pd.read_csv('temperature.csv')
file2_read = file2_read.drop(columns=['datetime', 'Beersheba', 'Tel Aviv District', 'Eilat', 'Haifa','Nahariyya', 'Jerusalem'])
file2_read = file2_read.dropna(axis=0)
file2_read = file2_read.apply(lambda x: x - 273.15)
file2_read = file2_read.mean(axis=0)
file2_read = pd.DataFrame({'City':file2_read.index, 'Temperature':file2_read.values})

#file2_read.columns = ["City", "Temperature"]
print("file2_read", file2_read)

file3_read = pd.read_csv('wind_speed.csv')
file3_read = file3_read.drop(columns=['datetime','Beersheba', 'Tel Aviv District', 'Eilat', 'Haifa','Nahariyya', 'Jerusalem'])
file3_read = file3_read.dropna(axis=0)
file3_read = file3_read.mean(axis=0)
file3_read = pd.DataFrame({'City':file3_read.index, 'Wind_Speed':file3_read.values})

#file3_read.columns = ["City", "Wind_Speed"]
print("file3_read", file3_read)

from functools import reduce
dfs = [file2_read, file3_read]
print()
print("file2_read", type(file2_read))
print("file3_read", type(file3_read))

cleaned_weather_data = reduce(lambda left,right: pd.merge(left,right,on=["City"]), dfs)
print(cleaned_weather_data)


# we load our  concatenate and prepared data
#input_data = np.asarray()
#clusters = KMeans(n_clusters=2, init=’k-means++’, max_iter=500, tol=0.0001).fit(input_data)

#we visualize the data and outputed clusters

X = cleaned_weather_data.iloc[:,1:]
print(X)
plt.scatter(X[:,0],X[:,1], label='True Position')  

from sklearn.cluster import KMeans
Kmean = KMeans(n_clusters=2)
#Kmean.fit(cleaned_weather_data)

