from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import numpy as np
from scipy import stats
import os
from bisect import bisect
import csv as csv


#____________inputs______________________
model_dir = os.curdir + '\\'
data_dir = os.curdir + '\\'

project = 'LGS'

projects = {
'DWR':{
'data_file':'DWQI_temp_2011-2015.csv',
'model_file':'two_45.opt',
'plot_name':'TOUT_Compare_DWR_2011-2015',
'title':'Dworshak Dam Tailwater (DWQI), 2011-2015'
},\
'LWG':{
'data_file':'LGNW_TDG_temp_2011-2015.csv',
'model_file':'two_43.opt',
'plot_name':'TOUT_Compare_LWG_2011-2015',
'title':'Lower Granite Dam Tailwater (LGNW), 2011-2015'
},\
'LGS':{
'data_file':'LGSW_TDG_temp_2011-2015.csv',
'model_file':'two_29.opt',
'plot_name':'TOUT_Compare_LGS_2011-2015',
'title':'Little Goose Dam Tailwater (LGSW), 2011-2015'
},\
'LMN':{
'data_file':'LMNW_TDG_temp_2011-2015.csv',
'model_file':'two_28.opt',
'plot_name':'TOUT_Compare_LMN_2011-2015',
'title':'Lower Monumental Dam Tailwater (LMNW), 2011-2015'
},\
'IHR':{
'data_file':'IDSW_TDG_temp_2011-2015.csv',
'model_file':'two_34.opt',
'plot_name':'TOUT_Compare_IHR_2011-2015',
'title':'Ice Harbor Dam Tailwater (IDSW), 2011-2015'
},\
'MCN':{
'data_file':'MCPW_TDG_temp_2011-2015.csv',
'model_file':'two_87.opt',
'plot_name':'TOUT_Compare_MCN_2011-2015',
'title':'McNary Dam Tailwater (MCPW), 2011-2015'
},\
'JDA':{
'data_file':'JHAW_TDG_temp_2011-2015.csv',
'model_file':'two_101.opt',
'plot_name':'TOUT_Compare_JDA_2011-2015',
'title':'John Day Dam Tailwater (JHAW), 2011-2015'
},\
'TDA':{
'data_file':'TDDO_TDG_temp_2011-2015.csv',
'model_file':'two_41.opt',
'plot_name':'TOUT_Compare_TDA_2011-2015',
'title':'The Dalles Dam Tailwater (TDDO), 2011-2015'
},\
'BON':{
'data_file':'CCIW_TDG_temp_2011-2015.csv',
'model_file':'two_76.opt',
'plot_name':'TOUT_Compare_BON_2011-2015',
'title':'Bonneville Dam Tailwater (CCIW), 2011-2015'
},\
'CHJ':{
'data_file':'CHQW_TDG_temp_2011-2015.csv',
'model_file':'two_42.opt',
'plot_name':'TOUT_Compare_CHJ_2011-2015',
'title':'Chief Joseph Dam Tailwater (CHQW), 2011-2015'
},\
'GCL':{
'data_file':'GCGW_TDG_Temp_2011-2015.csv',
'model_file':'two_317.opt',
'plot_name':'TOUT_Compare_GCL_PSU_2011-2015',
'title':'Grand Coulee Dam Tailwater (GCGW), 2011-2015'
},\
}

plot_name = projects[project]['plot_name']
data_file = projects[project]['data_file']
title = projects[project]['title']
model_file = projects[project]['model_file']
ylabel = 'Temperature ($^\circ$C)'
y2label = 'Temperature ($^\circ$F)'
key = ' '  #header in the data file for relevant temperature

#____________subroutines________________
#Reads in a temperature output file
#Returns a dictionary
#dictionary[jday][param] = result
#The params are temp_qwd, temp_str1, temp_str2, ...
#Works with version 3.7, USGS blending code
def two_read(filename, delimiter = 'fixed'):
    col_width = 8
    file_in = open(filename)
    #Dictionary jday:param:datum
    data_dict = {}
    #Ditch the header, not helpful
    header_lines = 3
    for i in range(header_lines):
        file_in.readline()
    for line in file_in:
        #splits line into columns and turns strings into floats
        #data_line = [float(line[x:x+col_width]) for x in range(0, len(line), col_width)]
        data_line = []
        try:
            strs
        except NameError:  #figure out how many strucutures there are but only do it once
            strs = []
            if delimiter == 'fixed':
##                str_num = len(line)/8 - 3
                str_num = len(line.split())-2
                print line
                print len(line), str_num
                for i in range(int(str_num)):
                    strs += ['temp_str' + str(i+1)]
            elif delimiter == 'comma':
                aaa = line.split(',')
                line1 = aaa[:-1]
                for i in range(len(line1[2:])):
                    strs += ['temp_str' + str(i+1)]
                print strs
        if delimiter == 'fixed':
            line = line.strip()
            line = line.split()
            for result in line:
                try: data_line.append(float(result))
                except ValueError:
                    data_line.append("NaN")
##            for x in range(0, len(line), col_width):
##                if x+col_width>len(line): continue
##                result = line[x:x+col_width]
##                try: data_line.append(float(result))
##                except ValueError:
##                    data_line.append("NaN")
        elif delimiter == 'comma':
            aaa = line.split(',')
            line1 = aaa[:-1]
            for result in line1:
                try: data_line.append(float(result))
                except ValueError:
                    data_line.append("NaN")
        #dictionary of each Jday
        jday = data_line[0]
        data_dict[jday]={}
        #assigns each column to the jday:param entry
        data_dict[jday]['temp_qwd'] = data_line[1]

##        for i in range(3,3+len(strs)):
        for i, struc in enumerate(strs):
            data_dict[jday][struc] = data_line[i+2]
    file_in.close()
    return data_dict

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
#This hopefully will be a good generic parser for csv and tab delimited files.
#Not programmed for fixed width yet.
#Assume first column is a datetime
#Assumes if time_fmt is not False, then time is the second column.
#returns data_dict[param][date] = result
def timeseries_parser(filein, skip = 0, dt_fmt = False, delimiter = ',', time_fmt = False):
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
#dictionary[jday][param] = result
model_result = two_read(model_dir + model_file, delimiter = 'comma')
#reorg
temp_dt = {}
for jday in model_result.keys():
    dt = datetime(year-1,12,31)+timedelta(jday)
    temp_dt[dt] = model_result[jday]['temp_qwd']
model_dt, model_wsel = format_dict_date_to_list_for_plot(temp_dt)

#Get measurements
#data_dict[head][jday] = result
temp_meas_jday = timeseries_parser(data_dir+data_file, skip = 2)
#def timeseries_parser(filein, skip = 0, dt_fmt = False, delimiter = ',', time_fmt = False):

temp_meas_dt = {}
for dt_string, result in temp_meas_jday['Temp(degC)'].iteritems():
    dt = datetime.strptime(dt_string, '%m/%d/%Y %H:%M')
    temp_meas_dt[dt] = result

meas_dt, meas_wsel = format_dict_date_to_list_for_plot(temp_meas_dt)

#Calculate the differnce model - measured
diffs = []
mod_vals = []
obs_vals = []

for dt in meas_dt:
    mod_date, distance = closest(dt,model_dt)
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

tstat_data = model_dir + 'TOUT_allStats_' + str(year) + '.csv'
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
ax1.plot(meas_dt, meas_wsel,label = 'Measured', linestyle='None', marker='.', color='r', markersize=1)
ax1.plot(model_dt, model_wsel, label = 'Modeled, RES-Ops Flows', color = 'k')
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2, frameon=False)
ax1.set_ylabel(ylabel)
ax1.yaxis.set_major_locator(MyLocator(7, prune='lower'))  #sets the max ticks at 7 and prunes the lower tick label
ax2 = ax1.twinx()
#make y-axes equivalent
y1min, y1max = ax1.get_ylim()
ax2.set_ylim(y1min*3.2808, y1max*3.2808)
ax2.set_ylabel(y2label)

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
ax1.set_title(title, y=1.09)

axx.plot(meas_dt, diffs,  linestyle='None', marker='.', color='k', markersize=1)
out_data = model_dir + 'T_model_minus_measured.csv'
graph_data_csv(out_data, meas_dt,diffs, header = ['datetime','bias'])
axx.set_ylabel('bias ($^\circ$C)')
#axx.set_xlabel(str(year))

#--------------------------------------
#THIS IS THE FIXED LINE (jwl 11/1/18)
x_range = [min(meas_dt), max(meas_dt)]
#THIS WAS THE PROBLEM LINE (It squished the data to not line up between plots.)
#x_range = axx.get_xlim()
#--------------------------------------

axx.plot(x_range, [0,0],color='k')
axx.text(0.05, 0.05, error_text, transform=axx.transAxes) #this places the legend

#axis ticks and labels
#axx.xaxis.set_major_locator(mdates.MonthLocator(interval=12))  This was coded (jwl 11/1/18)
#axx.xaxis.set_minor_locator(mdates.MonthLocator(interval=12))  This was coded (jwl 11/1/18)
axx.xaxis.set_minor_locator(ticker.IndexLocator(base=365.25,offset=180)) #revised to this (jwl 11/1/18)

#axx.xaxis.set_major_locator(mdates.MonthLocator())
#axx.xaxis.set_minor_locator(mdates.MonthLocator(bymonthday=15))
axx.xaxis.set_major_formatter(ticker.NullFormatter())
axx.xaxis.set_minor_formatter(mdates.DateFormatter('%Y'))
for tick in axx.xaxis.get_minor_ticks():
    tick.tick1line.set_markersize(0)
    tick.tick2line.set_markersize(0)
    tick.label1.set_horizontalalignment('center')


### This is just a test to see what is being plotted (Jim Lewis)
##MeasOut=np.column_stack((np.array(meas_dt),np.array(meas_wsel)))
##np.savetxt("Jim_meas.csv", MeasOut, delimiter=",", fmt='%s', header="x_meas, y_meas")
##ModelOut=np.column_stack((np.array(model_dt),np.array(model_wsel)))
##np.savetxt("Jim_model.csv", ModelOut, delimiter=",", fmt='%s', header="x_mod, y_mod")
##DiffOut=np.column_stack((np.array(meas_dt),np.array(diffs)))
##np.savetxt("Jim_diffs.csv", DiffOut, delimiter=",", fmt='%s', header="x_meas, y_diff")


plt.savefig(model_dir + plot_name +'.png', dpi = 200)
#plt.savefig(model_dir + plot_name +'.pdf')
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
axes.set_xlabel("Modeled Temperature, deg-C")
axes.set_ylabel("Measured Temperature, deg-C")
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


