import pandas as pd

data = pd.read_excel('data/Pearson_Edexcel_GCE_AS_and_AL_Mathematics_data_set_-_Issue_1_1.xls', sheet_name=1, header=5)

print(data.head())