import matplotlib.pyplot as plt 
import pandas as pd
from scipy.stats import spearmanr as spear
plt.style.use("seaborn-white")

data = pd.read_csv('PARSED_NW_Ground_Stations_2016.csv')
xs = data[['humidity','temperature','dew_point','pressure']]
y=  data['daily_precip']
labels={'humidity':'Humidity','temperature':'Temperature','dew_point':'Dew Point','pressure': 'Pressure'}
fig, ax = plt.subplots(2,2)

for idx,(name,values) in enumerate(xs.iteritems()):
    a = 0 if idx < 2 else 1
    ax[idx%2,a].scatter(values,y)
    print(name, spear(y, values))
    ax[idx%2,a].set_title('Distribution of Precipitation Against: ' + labels[name])
    ax[idx%2,a].set_xlabel(labels[name])
    ax[idx%2,a].set_ylabel('Daily Precipitation')
plt.show()