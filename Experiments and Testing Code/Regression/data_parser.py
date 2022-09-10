import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import math

#read data
data_read = "data/meteo_net/NW_Ground_Stations_2016.csv"
data_write = "data/meteo_net/DAY_PARSED_NW_Ground_Stations_2016.csv"

data = pd.read_csv(data_read)

data = data.drop(['lat','lon','height_sta'], axis=1)
data = data.sort_values(by=['number_sta', 'date'])
print(data.head())

dropables = []

i = [0,'2']
a = [[0,'2']]

for index, row in data.iterrows():    
    if row.isnull().values.any():        
        station = row['number_sta']
        date = row['date'][:8]
        drop = [station, date]        
        if drop not in dropables:            
            dropables.append(drop)
           

for each in dropables:
    print("Drop", len(data))
    data = data.drop(data['number_sta'] == each[0] & (data['date'].str.contains(each[1])))

exit()
data = data.dropna()
data = data.reset_index(drop=True)



#get list of all dates, ignore time 
date_list = [d[:8] for d in data['date'].unique()]
#get list of all stations
station_list = [s for s in data['number_sta'].unique()]

dates_ = []
for date in date_list:
    if date not in dates_:
        dates_.append(date)

date_list = dates_
stations = []
dates = []
windspd = []
winddir =[]
precip = []
hum = []
temp = []
dew = []
press = []
log_ = []
num_stations = len(station_list)
time_idx = 0#10*9#10 x hrs after midnight
#Combine readings to get the readings at the start of the day, and predict the total rainfall for a station over the whole day.
for idx, station in enumerate(station_list):
    remaining_stations = num_stations - idx
    print("%d remaining stations" % remaining_stations)
    for d, date in enumerate(date_list):
        remaining_dates = len(date_list) - d
        print("%d remaining dates" % remaining_dates)
        data_ = data.loc[(data['number_sta']==station) & (data['date'].str.contains(date))]
        if len(data_) > 0:
        #print(data_.iloc[time_idx, :])
            try:
                winddir.append(data_.iat[time_idx,2])
                windspd.append(data_.iat[time_idx,3])
                hum.append(data_.iat[time_idx,5])
                dew.append(data_.iat[time_idx,6])
                temp.append(data_.iat[time_idx,7])
                press.append(data_.iat[time_idx,8])
                
                sum_precip = sum(data_['precip'])
                precip.append(sum_precip)
                
                stations.append(station)
                dates.append(date)
            except:
                print(station,date)
                print(data_.iloc[time_idx, :])
                exit()


parsed_data = pd.DataFrame({"station":stations,
                            "date":dates,
                            "humidity":hum,
                            "temperature":temp,
                            "dew_point":dew,
                            "pressure":press,
                            "windspd":windspd,
                            "winddir":winddir,
                            "daily_precip":precip
                            })

print(parsed_data.head())

parsed_data.to_csv(data_write)