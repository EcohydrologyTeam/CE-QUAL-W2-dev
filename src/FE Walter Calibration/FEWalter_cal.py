#%%
import w2
import numpy as np
import pandas as pd
import sys
import os
from matplotlib import pyplot as plt
base_path = os.path.dirname(
os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.realpath(__file__)))))

w2_plotter_path = os.path.join(base_path, 'CE-QUAL-W2-dev', 'src')
sys.path.append(w2_plotter_path)
import w2
obs_path = os.path.join(base_path, 'CE-QUAL-W2-dev', 'obs_data_python')
observed_data = pd.read_excel(obs_path+'\\Reservoir_Wet_Chem_Walter_Beltz_2003-17-May-2010.xls')

w2_output_path = os.path.join(base_path, 'LehighW2', 'Models', 'WB1', '2001', 'V45 run')
#w2_output = pd.read_table(w2_output_path+'\\spr_wb1_wq_run86.opt')
w2_output = pd.read_table(r'C:\Users\b2edhijm\Documents\Projects\ERDC Steiss\FE Walter\LeheighW2\Models\WB1\2001\V45 run\spr_wb1_wq_run86.opt', delimiter = ",")
w2_output = w2_output.set_index('Constituent')

#%%
#### W2 OUTPUT HANDLING ####
print(len(w2_output.index[400]))
Index = 'Alkalinity'
print(len(Index))
add_space = 38 - len(Index)
for i in range(0,add_space):
    Index = Index+' '

print(Index)
w2_alk = w2_output.loc[Index]
print(w2_alk)
#%%
#### OBSERVED DATA HANDLING ####
observed_data = observed_data.set_index('STATION')

WA_2S = observed_data.loc['WA-2S']
WA_2M = observed_data.loc['WA-2M']
WA_2B = observed_data.loc['WA-2B']
WA_6S = observed_data.loc['WA-6S']
WA_6M = observed_data.loc['WA-6M']
WA_6B = observed_data.loc['WA-6B']
WA_7S = observed_data.loc['WA-7S']
WA_7M = observed_data.loc['WA-7M']
WA_7B = observed_data.loc['WA-7B']

WA_2 = pd.concat([WA_2S, WA_2M, WA_2B])
WA_6 = pd.concat([WA_6S, WA_6M, WA_6B])
WA_7 = pd.concat([WA_7S, WA_7M, WA_7B])

dates_2 = WA_2['JDAY'].tolist()
dates_2_round = [round(dates_2[i]) for i in range(0,len(dates_2))]
unique_dates = pd.Series(dates_2_round).value_counts().index.tolist()
date_series = pd.Series(dates_2_round)

alk_2 = WA_2['ALK'].tolist()
depth_2 = WA_2['DEPTH'].tolist()

count = len(unique_dates)
headers = [[] for i in range(count)]
alk2_by_date = [[] for i in range(count)]
depth2_by_date = [[] for i in range(count)]

for i in range(0,len(unique_dates)):
   #print(date_series[date_series == unique_dates[i]])
   headers[i].append(date_series[date_series == unique_dates[i]])
   for j in range(0,len(headers[i][0].index.tolist())):
        depth2_by_date[i].append(0 - depth_2[headers[i][0].index.tolist()[j]])
        alk2_by_date[i].append(alk_2[headers[i][0].index.tolist()[j]])


print(alk2_by_date)
print(depth2_by_date)

plt.scatter(alk2_by_date[0],depth2_by_date[0])

#%%
#plt.scatter(alk_2,depth_2)

##WA07 Alkalinity
#def grab_observed_variable(station_name: str, var_name: str):

#station = observed_data.index['WA_07B']
