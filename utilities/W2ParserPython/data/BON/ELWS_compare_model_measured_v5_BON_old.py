from __future__ import division

from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import os
from bisect import bisect
import csv as csv
import numpy as np
from scipy import stats

#import W2_routines as W2
#import data_download_format_metrics as ddfm

#____________inputs______________________
model_dir = os.curdir + '\\'
data_dir = os.curdir + '\\'

project = 'BON'

projects = {
'DWR':{
'data_file':'DWORSHAK-POOLELEV.csv',
'segment':'SEG  45',
'plot_name':'ELWS_Compare_DWR_2011-2015',
'title':'Dworshak Dam, 2011-2015'
},\
'LWG':{
'data_file':'LOWER GRANITE-POOLELEV.csv',
'segment':'SEG  43',
'plot_name':'ELWS_Compare_LWG_2011-2015',
'title':'Lower Granite Dam, 2011-2015'
},\
'LGS':{
'data_file':'LITTLE GOOSE-POOLELEV.csv',
'segment':'SEG  29',
'plot_name':'ELWS_Compare_LGS_2011-2015',
'title':'Little Goose Dam, 2011-2015'
},\
'LMN':{
'data_file':'LOWER MONUMENTAL-POOLELEV.csv',
'segment':'SEG  28',
'plot_name':'ELWS_Compare_LMN_2011-2015',
'title':'Lower Monumental Dam, 2011-2015'
},\
'IHR':{
'data_file':'ICE HARBOR-POOLELEV.csv',
'segment':'SEG  34',
'plot_name':'ELWS_Compare_IHR_2011-2015',
'title':'Ice Harbor Dam, 2011-2015'
},\
'MCN':{
'data_file':'MCNARY-POOLELEV.csv',
'segment':'SEG  87',
'plot_name':'ELWS_Compare_MCN_2011-2015',
'title':'McNary Dam, 2011-2015'
},\
'JDA':{
'data_file':'JOHN DAY-POOLELEV.csv',
'segment':'SEG 101',
'plot_name':'ELWS_Compare_JDA_2011-2015',
'title':'John Day Dam, 2011-2015'
},\
'TDA':{
'data_file':'THE DALLES-POOLELEV.csv',
'segment':'SEG  41',
'plot_name':'ELWS_Compare_TDA_2011-2015',
'title':'The Dalles Dam, 2011-2015'
},\
'BON':{
'data_file':'BONNEVILLE-POOLELEV.csv',
'segment':'SEG  76',
'plot_name':'ELWS_Compare_BON_2011-2015',
'title':'Bonneville Dam, 2011-2015'
},\
'CHJ':{
'data_file':'CHIEF JOSEPH-POOLELEV.csv',
'segment':'SEG  42',
'plot_name':'ELWS_Compare_CHQ_2011-2015',
'title':'Chief Joseph Dam, 2011-2015'
},\
'GCL':{
'data_file':'GRAND COULEE-POOLELEV.csv',
'segment':'SEG 317',
'plot_name':'ELWS_Compare_GCL_2011-2015',
'title':'Grand Coulee Dam, 2011-2015'
},\
}

plot_name = projects[project]['plot_name']
data_file = projects[project]['data_file']
title = projects[project]['title']
segment = projects[project]['segment']
model_file = 'wl.opt'
if project == 'GCL':
    model_file = 'tsr_81_seg317.opt'
ylabel = 'ELWS, NAVD88 (m)'
y2label = 'ELWS, NAVD88 (ft)'


#____________subroutines________________
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

#A routine that reads in the water surface elevations
#Returns dictionary, wsel[segment name][jday] = water sureface elevation
def wl_output2(filename):
    file_in = open(filename)
    #Dictionary jday:param:datum
    data_dict = {}
    header_line = file_in.readline()
    header_list = header_line.split(',')
    for seg in header_list[1:]:
        data_dict[seg]={}
    for line in file_in:
        #splits line into columns and turns strings into floats
        data_line = []
        line = line.rstrip()
        line_list = line.split(',')
        for result in line_list:
            try: data_line.append(float(result))
            except ValueError:
                data_line.append("NaN")
        #dictionary of each Jday
        jday = data_line[0]
        #assigns each column to the jday:param entry
        for index, head in enumerate(header_list[1:]):
            data_dict[head][jday] = data_line[index+1]
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

def stats_csv(fileOut, stats = [], header = []):
    with open(fileOut, 'wb') as f:
        writer = csv.writer(f)
        writer.writerows([header])
        writer.writerows([stats])

#_________code__________________

#Get the model results
year = get_year(model_dir)
if (project <> 'GCL'):
    #Returns dictionary, wsel[segment name][jday] = water sureface elevation
    wsel_all_jday = wl_output2(model_dir + 'wl.opt')
    wsel_dt = {}
    for jday, wsel in wsel_all_jday[segment].iteritems():
        date1 = datetime(year-1,12,31)+timedelta(jday)
        wsel_dt[date1] = wsel
    model_dt, model_wsel = format_dict_date_to_list_for_plot(wsel_dt)
    
    #Get measurements
    wsel_meas_jd = timeseries_parser(data_dir+data_file, skip = 2)
    wsel_meas_dt = {}
    for jday, wsel in wsel_meas_jd['ELEV'].iteritems():
        jday = float(jday)
        date1 = datetime(year-1,12,31)+timedelta(jday)
        wsel_meas_dt[date1] = wsel
    meas_dt, meas_wsel = format_dict_date_to_list_for_plot(wsel_meas_dt)
else:
    filein = model_dir+model_file
    with open(filein) as csvfile:
        tsr_in = csv.DictReader(csvfile)
        wsel_dt = {}
        for row in tsr_in:
            date1 = datetime(year-1,12,31)+timedelta(float(row["JDAY"]))
            wsel_dt[date1] = float(row["ELWS(m)"])
        model_dt, model_wsel = format_dict_date_to_list_for_plot(wsel_dt)
    
    #Get measurements
    ofilein=data_dir+data_file
    #header line is line 3; skip 2 lines
    with open(ofilein) as csvfile:
        for i in range(2):
            csvfile.next()
        obs_in = csv.DictReader(csvfile)
        wsel_meas_dt = {}
        for row in obs_in:
            date1 = datetime(year-1,12,31)+timedelta(float(row["JDAY"]))
            wsel_meas_dt[date1] = float(row["Water_surface_elev_m"])
        meas_dt, meas_wsel = format_dict_date_to_list_for_plot(wsel_meas_dt)

#Calculate the differnce model - measured
diffs = []
mod_vals = []
obs_vals = []

# open file for all stats
allStatsF = model_dir + 'ELWS_allStats_' + str(year) + '.csv'
allStats_csv = open(allStatsF,"a+")
allStats_csv.write('ME, MAE, RMSE, #OBS, R2, SLOPE, INTERCEPT, STD-ERR\n')

for dt in meas_dt:
    mod_date, distance = closest(dt,model_dt)
    meas = wsel_meas_dt[dt]
    mod = wsel_dt[mod_date]
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

elstat_data = model_dir + 'ELWS_allStats_' + str(year) + '.csv'
stats_csv(elstat_data, stats = [mean_err, abs_mean_err, rmse, len(diffs), r_value**2, slope, intercept, std_err], header = ['ME','MAE','RMSE', '#OBS', 'R2', 'SLOPE', 'INTERCEPT', 'STD-ERR'])

#This code is totatlly copied from somebody smarter than me
class MyLocator(ticker.MaxNLocator):
    def __init__(self, *args, **kwargs):
        ticker.MaxNLocator.__init__(self, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        return ticker.MaxNLocator.__call__(self, *args, **kwargs)

fig = plt.figure()
left, width = 0.13, 0.77
rect1 = [left, 0.50, width, 0.4]  #left, distance from bottom of graph (relative?), width, height (relative?)
rect2 = [left, 0.10, width, 0.4]
ax1 = fig.add_axes(rect1)
axx = fig.add_axes(rect2)
fig.autofmt_xdate()
ax1.plot(meas_dt, meas_wsel,label = 'RES-Ops',  linestyle='None', marker='.', color='r', markersize=1)
ax1.plot(model_dt, model_wsel, label = 'W2-modeled', color = 'k')

#ax1.plot(model_dt, , label = 'modeled')
#ax1.plot(meas_dt, ,label = 'measured')


ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2, frameon=False)
#ax1.legend()
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
out_data = model_dir + 'WSEL_model_minus_measured.csv'
graph_data_csv(out_data, meas_dt,diffs, header = ['datetime','bias'])
axx.set_ylabel('bias (m)')
#axx.set_xlabel(str(year))
x_range = axx.get_xlim()
axx.plot(x_range, [0,0],color='k')
axx.text(0.05, 0.05, error_text, transform=axx.transAxes)

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
axes.set_xlabel("Modeled ELWS, m NAVD88")
axes.set_ylabel("Measured ELWS, m NAVD88")
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
