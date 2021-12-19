import sys
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import matplotlib
from bisect import bisect
import numpy as np
from scipy import stats
import os
import csv as csv


#____________inputs______________________
model_dir = os.curdir + '\\'
data_dir = os.curdir + '\\'

project = 'LMN_tw'
startdate = datetime.strptime(''.join([str(2011),'-01-01']),'%Y-%m-%d').date()
enddate = datetime.strptime(''.join([str(2016),'-01-01']),'%Y-%m-%d').date()

projects = {
'LWG_tw':{
'gauge_type':'Sp',
'data_file':'LGNW_TDG_Temp_2011-2015.csv',
'mixed_outlet_file':'dwo_43.opt',
'spill_bays' : '''
dwo_gate2_seg43.opt
dwo_gate3_seg43.opt
dwo_gate4_seg43.opt
dwo_gate5_seg43.opt
dwo_gate6_seg43.opt
dwo_gate7_seg43.opt
dwo_gate8_seg43.opt
dwo_gate9_seg43.opt
''',
'plot_name':'TDG_TW_Compare_LWG_2011-2015',
'title':'Lower Granite Tailwater (LGNW) 2011-2015'
},\
'LWG_fb':{
'gauge_type':'Mix',
'data_file':'LWG_TDG_Temp_2011-2015.csv',
'mixed_outlet_file':'LWG_tsr_6_seg43.opt',
'plot_name':'TDG_FB_Compare_LWG_2011-2015',
'title':'Lower Granite Forebay TDG (LWG) 2011-2015'
},\
'LGS_tw':{
'gauge_type':'Mix',
'data_file':'LGSW_TDG_Temp_2011-2015.csv',
'mixed_outlet_file':'dwo_29.opt',
'spill_bays' : '''
dwo_gate2_seg29.opt
dwo_gate3_seg29.opt
dwo_gate4_seg29.opt
dwo_gate5_seg29.opt
dwo_gate6_seg29.opt
dwo_gate7_seg29.opt
dwo_gate8_seg29.opt
dwo_gate9_seg29.opt
dwo_gate10_seg29.opt
''',
'plot_name':'TDG_TW_Compare_LGS_2011-2015',
'title':'Little Goose Tailwater (LGSW) 2011-2015'
},\
'LGS_fb':{
'gauge_type':'Mix',
'data_file':'LGSA_TDG_Temp_2011-2015.csv',
'mixed_outlet_file':'LGS_tsr_6_seg29.opt',
'plot_name':'TDG_FB_Compare_LGS_2011-2015',
'title':'Little Goose Forebay (LGSA) 2011-2015'
},\
'LMN_tw':{
'gauge_type':'Sp',
'data_file':'LMNW_TDG_Temp_2011-2015.csv',
'mixed_outlet_file':'dwo_28.opt',
'spill_bays' : '''
dwo_gate2_seg28.opt
dwo_gate3_seg28.opt
dwo_gate4_seg28.opt
dwo_gate5_seg28.opt
dwo_gate6_seg28.opt
dwo_gate7_seg28.opt
dwo_gate8_seg28.opt
dwo_gate9_seg28.opt
''',
'plot_name':'TDG_TW_Compare_LMN_2011-2015',
'title':'Lower Monumental Tailwater (LMNW) 2011-2015'
},\
'LMN_fb':{
'gauge_type':'Mix',
'data_file':'LMNA_TDG_temp_2011-2015.csv',
'mixed_outlet_file':'LMN_tsr_6_seg28.opt',
'plot_name':'TDG_FB_Compare_LMN_2011-2015',
'title':'Lower Monumental Forebay (LMNA) 2011-2015'
},\
'IHR_tw':{
'gauge_type':'Mix',
'data_file':'IDSW_TDG_Temp_2011-2015.csv',
'mixed_outlet_file':'dwo_34.opt',
'spill_bays' : '''
dwo_gate2_seg34.opt
dwo_gate3_seg34.opt
dwo_gate4_seg34.opt
dwo_gate5_seg34.opt
dwo_gate6_seg34.opt
dwo_gate7_seg34.opt
dwo_gate8_seg34.opt
dwo_gate9_seg34.opt
dwo_gate10_seg34.opt
dwo_gate11_seg34.opt
''',
'plot_name':'TDG_TW_Compare_IHR_2011-2015',
'title':'Ice Harbor Tailwater (IDSW) 2011-2015'
},\
'IHR_fb':{
'gauge_type':'Mix',
'data_file':'IHRA_TDG_Temp_2011-2015.csv',
'mixed_outlet_file':'IHR_tsr_6_seg34.opt',
'plot_name':'TDG_FB_Compare_IHR_Temp_TDG_2011-2015',
'title':'Ice Harbor Forebay (IHRA) 2011-2015'
},\
'MCN_fb':{
'gauge_type':'Mix',
'data_file':'MCNA_TDG_Temp_2011-2015.csv',
'mixed_outlet_file':'MCN_tsr_6_seg87.opt',
'plot_name':'TDG_FB_Compare_MCN_Temp_TDG_2011-2015',
'title':'McNary Forebay (MCNA) 2011-2015'
},\
'MCN_tw':{
'gauge_type':'Mix',
'data_file':'MCPW_TDG_Temp_2011-2015.csv',
'mixed_outlet_file':'dwo_87.opt',
'spill_bays' : '''
dwo_gate10_seg87.opt
dwo_gate11_seg87.opt
dwo_gate12_seg87.opt
dwo_gate13_seg87.opt
dwo_gate14_seg87.opt
dwo_gate15_seg87.opt
dwo_gate16_seg87.opt
dwo_gate17_seg87.opt
dwo_gate18_seg87.opt
dwo_gate19_seg87.opt
dwo_gate20_seg87.opt
dwo_gate21_seg87.opt
dwo_gate22_seg87.opt
dwo_gate23_seg87.opt
dwo_gate2_seg87.opt
dwo_gate3_seg87.opt
dwo_gate4_seg87.opt
dwo_gate5_seg87.opt
dwo_gate6_seg87.opt
dwo_gate7_seg87.opt
dwo_gate8_seg87.opt
dwo_gate9_seg87.opt
''',
'plot_name':'TDG_TW_Compare_MCN_2011-2015',
'title':'McNary Tailwater (MCPW) 2011-2015'
},\
'JDA_fb':{
'gauge_type':'Mix',
'data_file':'JDY_TDG_Temp_2011-2015.csv',
'mixed_outlet_file':'JDA_tsr_2_seg101.opt',
'plot_name':'TDG_FB_Compare_JDA_Temp_TDG_2011-2015',
'title':'John Day Forebay (JDY) 2011-2015'
},\
'JDA_tw':{
'gauge_type':'Mix',
'data_file':'JHAW_TDG_Temp_2011-2015.csv',
'mixed_outlet_file':'dwo_101.opt',
'plot_name':'TDG_TW_Compare_JDA_2011-2015',
'title':'John Day Tailwater (JHAW) 2011-2015'
},\
'TDA_fb':{
'gauge_type':'Mix',
'data_file':'TDA_TDG_Temp_2011-2015.csv',
'mixed_outlet_file':'TDA_tsr_2_seg41.opt',
'plot_name':'TDG_FB_Compare_TDA_Temp_TDG_2011-2015',
'title':'The Dalles Forebay (TDA) 2011-2015'
},\
'TDA_tw':{
'gauge_type':'Mix',
'data_file':'TDDO_TDG_Temp_2011-2015.csv',
'mixed_outlet_file':'dwo_41.opt',
'plot_name':'TDG_TW_Compare_TDA_2011-2015',
'title':'The Dalles Tailwater (TDDO) 2011-2015'
},\
'BON_tw':{
'gauge_type':'Sp',
'data_file':'CCIW_TDG_Temp_2011-2015.csv',
'mixed_outlet_file':'dwo_76.opt',
'spill_bays' : '''
dwo_gate10_seg76.opt
dwo_gate11_seg76.opt
dwo_gate12_seg76.opt
dwo_gate13_seg76.opt
dwo_gate14_seg76.opt
dwo_gate15_seg76.opt
dwo_gate16_seg76.opt
dwo_gate17_seg76.opt
dwo_gate18_seg76.opt
dwo_gate19_seg76.opt
dwo_gate20_seg76.opt
dwo_gate2_seg76.opt
dwo_gate3_seg76.opt
dwo_gate4_seg76.opt
dwo_gate5_seg76.opt
dwo_gate6_seg76.opt
dwo_gate7_seg76.opt
dwo_gate8_seg76.opt
dwo_gate9_seg76.opt
''',
'plot_name':'TDG_TW_Compare_BON_2011-2015',
'title':'Bonneville Tailwater (CCIW) 2011-2015'
},\
'BON_fb':{'gauge_type':'Mix',
'data_file':'BON_TDG_Temp_2011-2015.csv',
'mixed_outlet_file':'BON_tsr_2_seg76.opt',
'plot_name':'TDG_FB_Compare_BON_Temp_TDG_2011-2015',
'title':'Bonneville Forebay (BON) 2011-2015'
},\
'CHJ_tw':{
'gauge_type':'Sp',
'data_file':'CHQW_TDG_Temp_2011-2015.csv',
'mixed_outlet_file':'dwo_42.opt',
'spill_bays' : '''
dwo_gate2_seg42.opt
''',
'plot_name':'TDG_TW_Compare_CHJ_Temp_TDG_2011-2015',
'title':'Chief Joseph Tailwater (CHQW) 2011-2015'
},\
'CHJ_fb':{'gauge_type':'Mix',
'data_file':'CHJ_TDG_Temp_2011-2015.csv',
'mixed_outlet_file':'CHJ_tsr_2_seg42.opt',
'plot_name':'TDG_FB_Compare_CHJ_Temp_TDG_2011-2015',
'title':'Chief Joseph Forebay (CHJ) 2011-2015'
},\
'GCL_fb':{'gauge_type':'Mix',
'data_file':'FDRW_TDG_Temp_2011-2015.csv',
'mixed_outlet_file':'tsr_44_seg317.opt',
'plot_name':'TDG_FB_Compare_GCL_PSU_2011-2015',
'title':'Grand Coulee Forebay (FDRW) 2011-2015'
},\
}


#LWG Forebay
#data_file = 'LWG_TDG_Temp_2014.csv'
#mixed_outlet_file = 'LWG_SystemModel_2014_NAVD88_DT_PEST_tsr_6_seg43.opt'

#LWG Tailwater
#data_file = 'LGNW_TDG_Temp_2014.csv'
#mixed_outlet_file = 'dwo_43.opt'
#spill_bays = '''
#dwo_gate2_seg43.opt
#dwo_gate3_seg43.opt
#dwo_gate4_seg43.opt
#dwo_gate5_seg43.opt
#dwo_gate6_seg43.opt
#dwo_gate7_seg43.opt
#dwo_gate8_seg43.opt
#dwo_gate9_seg43.opt
#'''

#BON forebay
#mixed_outlet_file = 'BON_TDG1_tsr_1_seg76.opt'
#data_file = 'BON_TDG_Temp_2014.csv'

#BON Tailwater
#data_file = 'CCIW_TDG_Temp_2014.csv'
#mixed_outlet_file = 'dwo_76.opt'
#spill_bays = '''
#dwo_gate2_seg76.opt
#dwo_gate3_seg76.opt
#dwo_gate4_seg76.opt
#dwo_gate5_seg76.opt
#dwo_gate6_seg76.opt
#dwo_gate7_seg76.opt
#dwo_gate8_seg76.opt
#dwo_gate9_seg76.opt
#dwo_gate10_seg76.opt
#dwo_gate11_seg76.opt
#dwo_gate12_seg76.opt
#dwo_gate13_seg76.opt
#dwo_gate14_seg76.opt
#dwo_gate15_seg76.opt
#dwo_gate16_seg76.opt
#dwo_gate17_seg76.opt
#dwo_gate18_seg76.opt
#dwo_gate19_seg76.opt
#'''

plot_name = projects[project]['plot_name']
data_file = projects[project]['data_file']
title = projects[project]['title']
mixed_outlet_file = projects[project]['mixed_outlet_file']
if 'spill_bays' in projects[project]:
    spill_bays = projects[project]['spill_bays']
    sb_files = spill_bays.split()


#_________subroutines__________________
def W2_generic_read(input_file, header_lines, col_width = False, header_names = None, param_first = False, print_errors=True, csv = False):
    #returns a dictionary[jday][param] = result
    #relies on robust header names

    file_in = open(input_file)
    #Dictionary jday:param:datum
    data_dict = {}
    data_dict_param = {}
    #Ditch the header
    line = file_in.readline()
    if line[0] == '$':
        csv = True
    for i in range(header_lines-1):
        print file_in.readline()
        line = file_in.readline()
    #Gets header lines, breaks into columns, removes whitespace
    if not header_names:
        header_names = []
        if csv == False:
            for x in range(0, len(line), col_width):
                name = line[x:x+col_width].strip()
                if name != "":
                    header_names.append(name)
        elif csv == True:
            aaa = line.strip()
            header_names = line.split(',')
    #print len(header_names), header_names
    for head in header_names:
        data_dict_param[head] = {}
    for line in file_in:
        #splits line into columns and turns strings into floats
        #data_line = [float(line[x:x+col_width]) for x in range(0, len(line), col_width)]
        data_line = []
        if csv == False:
            for x in range(0, len(line)-1, col_width):
                #Sometines need this next line, sometimes don't
                if x+col_width>len(line)-1:
                    result = line[x:]
                result = line[x:x+col_width]
                try: data_line.append(float(result))
                except ValueError:
                    if result != '\n' and print_errors:
                        print input_file, data_line, result, " not a number"
                    data_line.append("NaN")
        if csv == True:
            aaa = line.strip()
            bbb = line.split(',')
            for result in bbb:
                try: data_line.append(float(result))
                except ValueError:
                    if result != '\n' and print_errors:
                        print input_file, data_line, result, " not a number"
                    data_line.append("NaN")

        #dictionary of each Jday
        jday = data_line[0]
        data_dict[jday]={}
        #assigns each column to the jday:param entry
##        print jday, line, data_line
        for i in range(1,len(header_names)):
##            print i, jday,header_names[i]
            try:
                data_dict[jday][header_names[i]] = data_line[i]
                data_dict_param[header_names[i]][jday] = data_line[i]
            except IndexError:
                #The value is missing, so use previous jday value
                if print_errors:
                    print header_names, i, jday, data_line
                    print input_file + ' missing value on jday ' + str(jday) + ', using previous'
                jdays = data_dict.keys()
                jdays.sort()
                jday_last = jdays[-2]
                data_dict[jday][header_names[i]] = data_dict[jday_last][header_names[i]]
                data_dict_param[header_names[i]][jday] = data_dict_param[header_names[i]][jday_last]


    file_in.close()
    if param_first:
        return data_dict_param
    else:
        return data_dict

#This hopefully will be a good generic parser for csv and tab delimited files.
#Not programmed for fixed width yet.
#Assume first column is a datetime
#Assumes if time_fmt is not False, then time is the second column.
#returns data_dict[param][date] = result
def timeseries_parser(filein, skip = 0, dt_fmt = False, delimiter = ',', time_fmt = False):
    print 'timeseries_parser'
    print filein
    file_in = open(filein, 'r')
    #Dictionary param:date_text:datum
    data_dict = {}
    for s in range(skip):
        aaa = file_in.readline()
        print aaa
    header_line = file_in.readline()
    print header_line
    header_line = header_line.rstrip()
    header_list = header_line.split(delimiter)
    header_list.pop(0) #remove the first, usually a date
    header_list = [head.strip() for head in header_list]
    #sometimes there is a blank header at end
    if len(header_list[-1]) == 0:
        header_list.pop()
    print header_list
    for head in header_list:
        data_dict[head] = {}
    for line in file_in:
        #splits line into columns and turns strings into floats
        line = line.rstrip()  #strips white space and new line from the end of the string.
        line_list = line.split(delimiter)
        if dt_fmt and time_fmt == False:
            jday = datetime.strptime(line_list[0], dt_fmt)
        elif dt_fmt and time_fmt:
            combine_dt = line_list[0] + delimiter + line_list[1]
            if line_list[1][0:2] == '24':
                jday = datetime.strptime(line_list[0], dt_fmt)
                jday += timedelta(0.99)
            else:
                combine_fmt = dt_fmt + delimiter + time_fmt
                jday = datetime.strptime(combine_dt, combine_fmt)
        else:
            jday = line_list[0]
        #assigns each column to the jday:param entry
        for index, head in enumerate(header_list):
            try:
                data_dict[head][jday] = float(line_list[index+1])
            except ValueError:
                data_dict[head][jday] = line_list[index+1].strip()
    file_in.close()
    return data_dict

#function to take a dictionary[dates] = result and format into two lists, sorted by date, for plotting.
def format_dict_date_to_list_for_plot(indict, ignore = [-999], print_errors = True):
    dates = indict.keys()
    dates.sort()
    new_dates = []
    meas = []
    for date in dates:
        try:
            result = float(indict[date])
            if result in ignore: continue #skips to next loop if value in ignore list
            meas.append(result)
            new_dates.append(date)
        except IndexError:
            if print_errors == True:
                print 'Invalid Result, row = ', date, indict[date]
            continue
        except ValueError:
            if print_errors == True:
                print 'Value Error, row = ', date, indict[date]
            continue
    return new_dates, meas


#*************
#finding the closet value in a list -- useful for timeseries matching
#************
def closest(value, sorted_list_to_match):
    if value in sorted_list_to_match:
        closest_value = value
    elif value > sorted_list_to_match[-1]:
        closest_value = sorted_list_to_match[-1]
    elif value < sorted_list_to_match[0]:
        closest_value = sorted_list_to_match[0]
    else:
        closest_right = sorted_list_to_match[bisect(sorted_list_to_match, value)]
        closest_left = sorted_list_to_match[bisect(sorted_list_to_match, value)-1]
        if abs(value - closest_right) <= abs(value - closest_left):
            closest_value = closest_right
        else:
            closest_value = closest_left
    diff = closest_value - value
    return closest_value, diff

def Tc(Tf):
    return (5./9.)*(Tf-32)
def Tf(Tc):
    return (9./5.)*Tc+32

#write plot data to csv
def graph_data_csv(fileOut, xdata, ydata, header = []):
    zipped = zip(xdata,ydata)
    with open(fileOut, 'wb') as f:
        writer = csv.writer(f)
        writer.writerows([header])
        writer.writerows(zipped)
def get_year(path):
    filein = path + "w2_con.npt"
    W2_in = open(filein)
    for i in range(2000):
        j = W2_in.readline()
        j = j.split()
        if not j:   #blank line, skip
            continue
##        print j
        if j[0] == "TIME":
            nextline = W2_in.readline()
            nextline = nextline.split()
            year = int(nextline[2])
            break
    return year
def stats_csv(fileOut, stats = [], header = []):
    with open(fileOut, 'wb') as f:
        writer = csv.writer(f)
        writer.writerows([header])
        writer.writerows([stats])

#_________code__________________
#Get the model results
year = get_year(model_dir)

temp_dt = {}
if projects[project]['gauge_type'] == 'Sp':
    sb_dt = {}
    for sb in sb_files:
        model_result = W2_generic_read(model_dir + sb,2,csv = True)
        #reorg
        for jday in model_result.keys():
            if model_result[jday]['  TDG(%)'] == -99.0:
                continue
            dt = datetime(year-1,12,31)+timedelta(jday)
            if dt not in sb_dt:
                sb_dt[dt] = model_result[jday]['  TDG(%)']
    #get the mixed concentration also for when there is no spill
    model_result = W2_generic_read(model_dir+mixed_outlet_file, 2, 8, print_errors=True, csv = True)
    mix_dt = {}
    for jday in model_result.keys():
        dt = datetime(year-1,12,31)+timedelta(jday)
        mix_dt[dt] = model_result[jday]['  TDG(%)']
        if dt not in sb_dt:
            sb_dt[dt] = model_result[jday]['  TDG(%)']
    temp_dt = sb_dt
    model_dt, model_wsel = format_dict_date_to_list_for_plot(temp_dt)


if projects[project]['gauge_type'] == 'Mix':
    #dictionary[jday][param] = result
    if 'dwo' in mixed_outlet_file:
        model_result = W2_generic_read(model_dir+mixed_outlet_file, 2, 8, print_errors=True, csv = True)
    elif 'tsr' in mixed_outlet_file:
        model_result = W2_generic_read(model_dir+mixed_outlet_file, 0, print_errors=True, csv = True)
    #reorg

    for jday in model_result.keys():
        dt = datetime(year-1,12,31)+timedelta(jday)
        temp_dt[dt] = model_result[jday]['  TDG(%)']
    model_dt, model_wsel = format_dict_date_to_list_for_plot(temp_dt)


##model_file = 'tsr_1_seg41.opt'
##data_file = 'TDA_TDG_Temp.csv'
##plot_name = 'TDG_forebay_compare'

##model_file = 'tsr_2_seg2.opt'
##data_file = 'JHAW_TDG_Temp.csv'
##plot_name = 'TDG_headwater_compare'


ylabel = 'TDG (% sat)'
y2label = ''

key = 'TDG(%sat)'  #header in the data file for relevant temperature




#Get measurements
#data_dict[head][jday] = result
temp_meas_jday = timeseries_parser(data_dir+data_file, skip = 2)
temp_meas_dt = {}
for dt_string, result in temp_meas_jday[key].iteritems():
##    dt = datetime.strptime(dt_string, '%m/%d/%Y %H:%M')
#    dt = datetime(year-1,12,31)+timedelta(float(dt_string))
    try:
        dt = datetime.strptime(dt_string, '%m/%d/%Y %H:%M')
    except ValueError:
        dt = datetime.strptime(dt_string, '%m/%d/%Y')
    temp_meas_dt[dt] = result

meas_dt, meas_wsel = format_dict_date_to_list_for_plot(temp_meas_dt)

#Calculate the differnce model - measured
diffs = []
mod_vals = []
obs_vals = []
#Skip first model_dt because the output is garbage
for dt in meas_dt:
    mod_date, distance = closest(dt,model_dt[1:])
    meas = temp_meas_dt[dt]
    mod = temp_dt[mod_date]
    diffs += [mod-meas]
    mod_vals += [mod]
    obs_vals += [meas]
mean_err = np.mean(diffs)
abs_mean_err = np.mean([abs(diff) for diff in diffs])
aaa = np.mean([diff*diff for diff in diffs])
rmse = np.sqrt(aaa)

# Calculate other stats using scipy library.
slope, intercept, r_value, p_value, std_err = stats.linregress(mod_vals,obs_vals)

error_text = 'ME=' + "{:.2f}".format(mean_err)
error_text += ', MAE=' + "{:.2f}".format(abs_mean_err)
error_text += ', RMSE=' + "{:.2f}".format(rmse)


lin_text = 'y = '+"{:.2f}".format(slope) + 'x + ' + "{:.2f}".format(intercept)
lin_text += ', R-sq = ' + "{:.2f}".format(r_value**2)
lin_text += ', N = ' + "{:g}".format(len(diffs))

tstat_data = model_dir + 'TDG_FB_allStats_' + str(year) + '.csv'
stats_csv(tstat_data, stats = [mean_err, abs_mean_err, rmse, len(diffs), r_value**2, slope, intercept, std_err], header = ['ME','MAE','RMSE', '#OBS', 'R2', 'SLOPE', 'INTERCEPT', 'STD-ERR'])

#This code is totatlly copied from somebody smarter than me
class MyLocator(ticker.MaxNLocator):
    def __init__(self, *args, **kwargs):
        ticker.MaxNLocator.__init__(self, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        return ticker.MaxNLocator.__call__(self, *args, **kwargs)

fig = plt.figure()
left, width = 0.13, 0.77
rect1 = [left, 0.45, width, 0.35]  #left, distance from bottom of graph (relative?), width, height (relative?)
rect2 = [left, 0.10, width, 0.35]
ax1 = fig.add_axes(rect1)
axx = fig.add_axes(rect2)
fig.autofmt_xdate()
ax1.plot(meas_dt, meas_wsel,label = 'Measured TDG, %Sat', linestyle='None', marker='.', color='r', markersize=1)
ax1.plot(model_dt[1:], model_wsel[1:], label = 'Modeled TDG, %Sat', color = 'k')
ax1.legend(loc= 'center', bbox_to_anchor=(0.5, 1.15), ncol = 2, frameon=False)
ax1.set_ylabel(ylabel)
ax1.yaxis.set_major_locator(MyLocator(7, prune='lower'))  #sets the max ticks at 7 and prunes the lower tick label


#axis ticks and labels
MonthDayFmt = mdates.DateFormatter('%Y')
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=12))
ax1.xaxis.set_minor_locator(mdates.MonthLocator(interval=12))
ax1.xaxis.set_major_formatter(ticker.NullFormatter())
ax1.xaxis.set_minor_formatter(mdates.DateFormatter('%Y'))
for tick in ax1.xaxis.get_minor_ticks():
    tick.tick1line.set_markersize(0)
    tick.tick2line.set_markersize(0)
    tick.label1.set_horizontalalignment('center')
    tick.set_visible(False)
##ax2.xaxis.set_ticklabels([],visible = False)
ax1.set_title(title, y=1.3)

axx.plot(meas_dt, diffs, linestyle='None', marker='.', color='k', markersize=1)
axx.plot(ax1.get_xlim(),[0,0], color='k')
##ymin, ymax = axx.get_ylim()
##axx.set_ylim(min(diffs),ymax)
out_data = model_dir + plot_name + '_bias.csv'
graph_data_csv(out_data, model_dt[1:],diffs, header = ['datetime','bias'])
axx.set_ylabel('TDG Bias (%Sat)')
##axx.set_xlabel(str(year))

##axx.text(0.05, 0.05, error_text)
axx.text(0.05, 0.05, error_text, transform=axx.transAxes)
axx.set_xlim(ax1.get_xlim())

#axis ticks and labels
axx.xaxis.set_major_locator(mdates.MonthLocator(interval=12))
axx.xaxis.set_minor_locator(mdates.MonthLocator(interval=12))
#axx.xaxis.set_major_locator(mdates.MonthLocator())
#axx.xaxis.set_minor_locator(mdates.MonthLocator(bymonthday=15))
axx.xaxis.set_major_formatter(ticker.NullFormatter())
axx.xaxis.set_minor_formatter(mdates.DateFormatter('%Y'))
for tick in axx.xaxis.get_minor_ticks():
    tick.tick1line.set_markersize(0)
    tick.tick2line.set_markersize(0)
    tick.label1.set_horizontalalignment('center')


plt.savefig(model_dir + plot_name +'.png', dpi = 200)
plt.close()
#os.startfile(model_dir + plot_name +'.png', 'open')

fig = plt.figure()
plt.plot(mod_vals, obs_vals,linestyle='None', marker='.', color='k', markersize=1, label="Mod-Obs Data Pairs")
#fit_vals = intercept + (slope*mod_vals)
fit_vals = []
for v in mod_vals:
    fval = float(slope)*v + float(intercept)
    fit_vals += [fval]
plt.plot(mod_vals, fit_vals,'r',label="Linear Trendline")
#adjust axes
axes = plt.gca()
axes.set_aspect('equal','datalim')
axes.axis('square')
axes.set_xlabel("Modeled TDG, %Sat")
axes.set_ylabel("Measured TDG, %Sat")
ymin,ymax = axes.get_ylim()
xmin,xmax = axes.get_xlim()
#title and text
plt.title(title)
plt.legend(loc='best')
axes.text(xmax, ymin, lin_text, horizontalalignment='right', verticalalignment='bottom')


name = plot_name +'_linearFit' +'.png'
plt.savefig(model_dir + name, dpi = 200)
#    name2 = plot_name + '_' + str(model_info['depth']) +'_m_linearFit' +'.pdf'
#    plt.savefig(model_dir + name2)
plt.close()



