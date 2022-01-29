import pandas as pd
from matplotlib import pyplot as plt
import datetime
import seaborn as sns
import h5py
import warnings
from enum import Enum
warnings.filterwarnings("ignore")

plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = (15, 9)
plt.rcParams['grid.color'] = '#E0E0E0'
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['axes.facecolor'] = '#FBFBFB'
plt.rcParams["axes.edgecolor"] = '#222222'
plt.rcParams["axes.linewidth"] = 0.5
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['figure.subplot.hspace'] = 0.05  # Shrink the horizontal space

# Custom curve colors
# Using mountain and lake names for new color palettes
rainbow = ['#3366CC', '#0099C6', '#109618', '#FCE030', '#FF9900', '#DC3912']  # (blue, teal, green, yellow, orange, red)
everest = ['#3366CC', '#DC4020', '#10AA18', '#0099C6', '#FCE030', '#FF9900', ]  # (blue, red, green, teal, yellow, orange)

k2 = (
    sns.color_palette('husl', desat=0.8)[4], # blue
    sns.color_palette('tab10')[3], # red
    sns.color_palette('deep')[2], # green
    sns.color_palette('tab10', desat=0.8)[1], # purple
    sns.color_palette('deep', desat=0.8)[4], # purple
    sns.color_palette('colorblind')[2], # sea green
    sns.color_palette('colorblind')[0], # deep blue
    sns.color_palette('husl')[0], # light red
)


# Define string formatting constants, which work in string format statements
DEG_C_ALT = u'\N{DEGREE SIGN}C'

# Define default line color
DEFAULT_COLOR = '#4488ee'


class FileType(Enum):
    unknown = 0
    fixed_width = 1
    csv = 2


def round_time(dt: datetime.datetime = None, roundTo=60):
    '''
    Round a datetime object to any time in seconds

    dt : datetime.datetime object
    roundTo : Closest number of seconds to round to. Default = 1 minute.
    '''
    if dt == None:
        dt = datetime.datetime.now()
    seconds = (dt.replace(tzinfo=None) - dt.min).seconds
    rounding = (seconds+roundTo/2) // roundTo * roundTo
    return dt + datetime.timedelta(0, rounding-seconds, -dt.microsecond)


def day_of_year_to_datetime(year: int, day_of_year_list: list):
    '''
    Convert a list of day-of-year values to datetime objects

    year : int
        Start year of the data
    day_of_list : list
        List of day-of-year values, e.g., from CE-QUAL-W2
    '''
    day1 = datetime.datetime(year, 1, 1, 0, 0, 0)
    datetimes = []
    for d in day_of_year_list:
        # Compute the difference, subtracting 1 from the day_of_year
        dx = day1 + datetime.timedelta(days=(d-1))
        # Round the time
        dx = round_time(dt=dx, roundTo=60*60)
        datetimes.append(dx)
    return datetimes


def dataframe_to_date_format(year: int, data_frame: pd.DataFrame):
    '''
    Convert the day-of-year column in a CE-QUAL-W2 data frame
    to datetime objects

    year : int
        Start year of the data
    data_frame : pandas.DataFrame object
        Data frame to convert
    '''
    datetimes = day_of_year_to_datetime(year, data_frame.index)
    data_frame.index = datetimes
    data_frame.index.name = 'Date'
    return data_frame


def read_npt(infile: str, year: int, data_columns: list[str], skiprows: int = 3):
    '''Read CE-QUAL-W2 time series (fixed-width format, *.npt files)'''

    ncols_to_read = len(data_columns) + 1  # number of columns to read, including the date/day column
    columns_to_read = ['DoY', *data_columns]
    return pd.read_fwf(infile, skiprows=skiprows, widths=ncols_to_read*[8], names=columns_to_read, index_col=0)


def read_csv(infile: str, year: int, data_columns: list[str], skiprows: int = 3):
    '''Read CE-QUAL-W2 time series (CSV format)'''

    try:
        df = pd.read_csv(infile, skiprows=skiprows, names=data_columns, index_col=0)
    except:
        # Handle trailing comma, which adds an extra (empty) column
        df = pd.read_csv(infile, skiprows=skiprows, names=[*data_columns, 'JUNK'], index_col=0)
        df = df.drop(axis=1, labels='JUNK')
    return df


def read(infile: str, year: int, data_columns: list[str], skiprows: int = 3, file_type: FileType = None):
    '''
    Read CE-QUAL-W2 time series data (npt and csv formats) and convert the Day of Year (Julian Day) to date-time format

    This function automatically detects the file type, if the file is named with *.npt or *.csv extensions. 
    '''

    # If not defined, set the file type using the input filename
    if not file_type:
        if infile.lower().endswith('.csv'):
            file_type = FileType.csv
        elif infile.lower().endswith('.npt'):
            file_type = FileType.fixed_width
        else:
            raise Exception('The file type was not specified, and it could not be determined from the filename.')

    # Read the data
    if file_type == FileType.fixed_width:
        df = read_npt(infile, year, data_columns, skiprows=skiprows)
    elif file_type == FileType.csv:
        df = read_csv(infile, year, data_columns, skiprows=skiprows)
    else:
        raise Exception('Error: file_type is not defined correctly.')

    # Convert day-of-year column of the data frames to date format
    df = dataframe_to_date_format(year, df)

    return df


def read_met(infile: str, year: int, data_columns: list[str] = None, skiprows: int = 3):
    '''Read meteorology time series'''
    if not data_columns:
        data_columns = [
            'Air Temperature ($^oC$)',
            'Dew Point Temperature ($^oC$)',
            'Wind Speed (m/s)',
            'Wind Direction (radians)',
            'Cloudiness (fraction)',
            'Solar Radiation ($W/m^2$)'
        ]

    return read(infile, year, data_columns, skiprows=skiprows)


def get_colors(df: pd.DataFrame, palette: str, min_colors=6):
    '''Get list of colors from the specified Seaborn color palette'''

    colors = sns.color_palette(palette, min(min_colors, len(df.columns)))
    return colors


def plot(df, title: str = None, legend_list: list[str] = None,
         xlabel: str = None, ylabel: str = None, colors: list[str] = None,
         figsize=(15, 9), style: str = '-', palette: str = 'colorblind', **kwargs):
    '''Plot all columns in one figure'''

    fig, axes = plt.subplots(figsize=figsize)

    if not colors:
        colors = get_colors(df, palette, min_colors=6)

    axes.set_prop_cycle("color", colors)

    df.plot(ax=axes, title=title, ylabel=ylabel, style=style)

    if legend_list:
        axes.legend(legend_list)

    fig.tight_layout()  # This resolves a lot of layout issues
    return fig


def multiplot(df, title: str = None, legend_list: list[str] = None, xlabel: str = None,
              ylabels: list[str] = None, colors: list[str] = None, figsize=(15, 21),
              style: str = '-', palette: str = 'colorblind', **kwargs):
    '''Plot each column as a separate subplot'''

    fig, axes = plt.subplots(figsize=figsize)
    plt.subplots_adjust(top=0.97)  # Save room for the plot title

    if not colors:
        colors = get_colors(df, palette, min_colors=6)

    axes.set_prop_cycle("color", colors)

    subplot_axes = df.plot(subplots=True, ax=axes, sharex=True, legend=False, title=title, style=style, color=colors)

    if title:
        axes.set_title(title)

    if not ylabels:
        ylabels = df.columns

    # Label each sub-plot's y-axis
    for ax, ylabel in zip(subplot_axes, ylabels):
        ax.set_ylabel(ylabel)

    if legend_list:
        axes.legend(legend_list)

    fig.tight_layout()  # This resolves a lot of layout issues
    return fig


def write_hdf(df: pd.DataFrame, group: str, outfile: str, overwrite=True):
    '''
    Write CE-QUAL-W2 timeseries dataframe to HDF5

    The index column must be a datetime array. This columns will be written to HDF5 as a string array. 
    Each data column will be written using its data type.
    '''

    with h5py.File(outfile, 'a') as f:
        index = df.index.astype('str')
        string_dt = h5py.special_dtype(vlen=str)
        date_path = group + '/' + df.index.name
        if overwrite and (date_path in f):
            del f[date_path]
        f.create_dataset(date_path, data=index, dtype=string_dt)

        for col in df.columns:
            ts_path = group + '/' + col
            if overwrite and (ts_path in f):
                del f[ts_path]
            f.create_dataset(ts_path, data=df[col])


def read_hdf(group: str, infile: str, variables: list[str]):
    '''
    Read CE-QUAL-W2 timeseries dataframe to HDF5

    This function assumes that a string-based datetime array named Date is present. This will be read and 
    assiened as the index column of the output pandas dataframe will be a datetime array. 
    '''

    with h5py.File(infile, 'r') as f:
        # Read dates
        date_path = group + '/' + 'Date'
        dates_str = f.get(date_path)

        # Read time series data
        ts = {}
        for v in variables:
            ts_path = group + '/' + v
            ts[v] = f.get(ts_path)

        dates = []
        for dstr in dates_str:
            dstr = dstr.decode('utf-8')
            d = pd.to_datetime(dstr)
            dates.append(d)
        
        df = pd.DataFrame(ts, index=dates)
        return df