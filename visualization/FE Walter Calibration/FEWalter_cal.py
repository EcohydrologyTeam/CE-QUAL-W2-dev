import w2
import numpy as np
import pandas as pd
import sys
import os
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
#print(observed_data)

w2_output_path = os.path.join(base_path, 'LehighW2', 'Models', 'WB1', '2001', 'V45 run')
#w2_output = pd.read_table(w2_output_path+'\\spr_wb1_wq_run86.opt')
w2_output = pd.read_table(r'C:\Users\b2edhijm\Documents\Projects\ERDC Steiss\FE Walter\LeheighW2\Models\WB1\2001\V45 run\spr_wb1_wq_run86.opt', delimiter = ",")


print(observed_data.columns.tolist()[0])

#for i in range(w2_output.columns):
    #print(w2_output.columns[i])
#print(w2_output[:,1])