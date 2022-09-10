from os import stat
from scipy.stats.stats import alexandergovern
from scipy.stats.stats import f_oneway
import torch
import matplotlib.pyplot as plt 
import matplotlib
import pickle 
import pandas as pd
import statistics
import scipy.stats as st
import numpy as np
plt.style.use("seaborn-white")




def line_plot(data: list, labels,title=None, xl=None, yl=None, xt=None,yt=None, is_dotted=None):
    fig, ax = plt.subplots()
    if not is_dotted:
        is_dotted = [0 for _ in range(len(data))]
    for idx,each in enumerate(data):
            if is_dotted[idx]==1:
                ax.plot(each, '--',label=labels[idx], color='black')
            else:        
                x = [q for q in range(len(each))]
                plt.plot(each, label=labels[idx])
    if title:
        ax.set_title(title)
    if xl:
        ax.set_xlabel(xl)
    if yl:
        ax.set_ylabel(yl)
    if xt:
        ax.set_xticks(xt)
    if yt:
        ax.set_yticks(yt)
    plt.legend()
    plt.show()

def get_cum_acc(data):
    acc = []
    total = 0
    correct = 0
    for each in data:
        total += 1
        correct += 1 if each == 1 else 0
        acc.append(correct/total)
    return acc


with open('ewc_acc_history.pickle', 'rb') as f:
    ewc_history = pickle.load(f)

with open('lstm_acc_history.pickle', 'rb') as f:
    lstm_history = pickle.load(f)
print(ewc_history)
data1 = ewc_history
data2 = lstm_history
fig,ax = plt.subplots()
cols=['red','green']
x = [x for x in range(len(data1))]
y = [0.35 for _ in range(len(data1))]
ax.scatter(x,y, c=data1,cmap=matplotlib.colors.ListedColormap(cols))

x = [x for x in range(len(data2))]
y = [0.65 for _ in range(len(data2))]
ax.scatter(x,y, c=data2,cmap=matplotlib.colors.ListedColormap(cols))
ax.legend(())



ax.set_xlabel('Day')
ax.set_xticks([x for x in range(len(data1))])
ax.set_yticks([0,0.35, 0.65, 1])
ax.set_yticklabels(['','EWC+LSTM', 'LSTM', ''])
ax.set_ylabel('Model')
ax.set_title('Accuracy of Predictions From LSTM and EWC + LSTM')
plt.show()



ewc_acc = get_cum_acc(ewc_history)
lstm_acc = get_cum_acc(lstm_history)
line_plot([ewc_acc, lstm_acc],['ewc','lstm'],title='Cumulative Accuracy by Day',xl='Day',yl='Total Acc', xt=[x for x in range(len(ewc_acc))])

soil_data = pd.read_csv('../data/cambridge_sensor_data.csv')
labels = {'EWC': 'ewc_sm','LSTM':'lstm_sm','Sensor':'sen_sm','Human':'ctrl_sm'}

soil_data = soil_data[list(labels.values())]
soil_data = soil_data.drop([x for x in range(10)])
soil_data.reset_index(inplace=True)
data = []
for each in labels.keys():
    data.append(list(soil_data[labels[each]]))



extra_labels=['Min Optimal','Max Optimal']
is_dotted = [0 for _ in range(len(labels.keys()))]
is_dotted.extend([1 for _ in range(len(extra_labels))])

bins = np.linspace(200, 850, 20)
xticks = np.arange(200,850,50)

plt.hist(data, label=list(labels.keys()), bins=bins)
plt.legend()
plt.xticks(xticks)
plt.xlabel('Soil Moisture')
plt.ylabel('Frequency')
plt.title('Soil-Moisture Level Frequency Across Entire Observation Period')
plt.show()


labs = list(labels.keys())

fig, ax = plt.subplots()
dic = dict(labels)
print([x for x in range(len(dic.keys()))])
#ax.set_xticks([x for x in range(len(dic.keys()))])
ax.set_yticklabels(list(dic.keys()))
ax.boxplot(data, vert=False)
ax.set_title("Distribution of Soil-Moisture Values Per Method")
ax.set_ylabel('Method')
ax.set_xlabel('Soil Moisture Level Across Entire Observation Period')

plt.show()

fig, ax = plt.subplots()
ewc_labs = ['Full Timeframe','First Seven Days','Last 7 Days']
print([x for x in range(len(ewc_labs))])
#ax.set_xticks([x for x in range(len(dic.keys()))])

ewc_data = data[0]
ewc_box = [ewc_data, ewc_data[:7],ewc_data[-7:]]
ax.set_yticklabels(ewc_labs)
ax.boxplot(ewc_box, vert=False)
ax.set_title("Distribution of Soil-Moisture Values For EWC Over Time")
ax.set_ylabel('Method')
ax.set_xlabel('Soil Moisture Level')

plt.show()
print(dic)

p_value = alexandergovern(data[0],data[1],data[2],data[3])
print(p_value)
def split(data):
    return (data[:7], data[-7:])


for each in data:    
    mean = statistics.mean(each)
    mode = statistics.mode(each)
    median = statistics.median(each)
    stdev = statistics.stdev(each)
    skew = st.skew(each)
    kurt = st.kurtosis(each)
    stri = 'Mean: %.3f\nMedian: %.3f\nStdDev: %.3f\nSkew: %.3f \nKurtosis: %.3f' % (mean,median,stdev,skew,kurt)
    print(stri)
    print('\n\n')
# for each in data:
#     d = split(each)
#     mean = (statistics.mean(d[0]),statistics.mean(d[1]))
#     mode = (statistics.mode(d[0]),statistics.mode(d[1]))
#     median = (statistics.median(d[0]),statistics.median(d[1]))
#     stdev = (statistics.stdev(d[0]),statistics.stdev(d[1]))
#     skew = (st.skew(d[0]), st.skew(d[1]))
#     kurt = (st.kurtosis(d[0]),st.kurtosis(d[1]))
#     stri = 'Mean: %.3f %.3f\nMedian: %.3f  %.3f\nStdDev: %.3f  %.3f\nSkew: %.3f  %.3f\nKurtosis: %.3f  %.3f' % (mean[0],mean[1],median[0],median[1],stdev[0],stdev[1],skew[0],skew[1],kurt[0],kurt[1])
#     print(stri)
#     print('\n\n')
line_plot(data, labels=labs, title="Soil-Moisture Value Over Time", xl="Day", yl="Soil Moisture Value", xt=[x for x in range(len(data[0]))], is_dotted=is_dotted)

with open('pred_history.pickle', 'rb') as f:
    pred = pickle.load(f)
    for idx in range(len(pred)):
        pred[idx] = (pred[idx][0][1],pred[idx][1][1])
data1 = [o[0] for o in pred]
data2 = [o[1] for o in pred]
fig,ax = plt.subplots()
cols=['blue','orange']
x = [x for x in range(len(data1))]
y = [0.35 for _ in range(len(data1))]
ax.scatter(x,y, c=data1,label=data1,cmap=matplotlib.colors.ListedColormap(cols))

x = [x for x in range(len(data2))]
y = [0.65 for _ in range(len(data2))]
ax.scatter(x,y, c=data2,cmap=matplotlib.colors.ListedColormap(cols))
ax.legend(())



ax.set_xlabel('Day')
ax.set_xticks([x for x in range(len(data1))])
ax.set_yticks([0,0.35, 0.65, 1])
ax.set_yticklabels(['','EWC+LSTM', 'LSTM', ''])
ax.set_ylabel('Model')
ax.set_title('Daily Predictions From LSTM and EWC + LSTM')
plt.show()