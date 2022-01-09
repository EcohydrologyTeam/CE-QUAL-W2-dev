import pandas
from matplotlib import pyplot as plt
import datetime
import seaborn as sns
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
custom_colors = ['#3366CC', '#0099C6', '#109618', '#FCE030',
                 '#FF9900', '#DC3912']  # (blue, teal, green, yellow, orange, red)

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


def dataframe_to_date_format(year: int, data_frame: pandas.DataFrame):
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


def read(infile: str, year: int, data_columns: list[str],
         skiprows: int = 3, file_type: FileType = None):
    '''Read CE-QUAL-W2 time series data (npt and csv formats)'''

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
        ncols_to_read = len(data_columns) + 1  # number of columns to read, including the date/day column
        columns_to_read = ['DoY', *data_columns]
        df = pandas.read_fwf(infile, skiprows=skiprows, widths=ncols_to_read*[8], names=columns_to_read, index_col=0)
    elif file_type == FileType.csv:
        try:
            df = pandas.read_csv(infile, skiprows=skiprows, names=data_columns, index_col=0)
        except:
            # Handle trailing comma, which adds an extra (empty) column
            df = pandas.read_csv(infile, skiprows=skiprows, names=[*data_columns, 'JUNK'], index_col=0)
            df = df.drop(axis=1, labels='JUNK')
    else:
        raise Exception('Error: file_type is not defined correctly.')

    # Convert day-of-year column of the data frames to date format
    df = dataframe_to_date_format(year, df)

    return df


def read_met(infile: str, year: int, data_columns: list[str] = None, skiprows: int = 3):
    '''Read meteorology time series'''
    if not data_columns:
        data_columns = [
            f'Air Temperature (^oC)',
            f'Dew Point Temperature (^oC)',
            f'Wind Speed (m/s)',
            f'Wind Direction (radians)',
            f'Cloudiness (fraction)',
            f'Solar Radiation ($W/m^2$)'
        ]

    return read(infile, year, data_columns, skiprows=skiprows)


def plot(df, title: str = None, legend_list: list[str] = None, xlabel: str = None, ylabel: str = None,
         colors=custom_colors, figsize=(15, 10), multi_plot_figsize=(15, 21),
         line_color=DEFAULT_COLOR, multiplot: bool = False, style: str = '-', **kwargs):
    if multiplot:
        fig, axes = plt.subplots(figsize=multi_plot_figsize)
        plt.subplots_adjust(top=0.97)  # Save room for the plot title
        axes.set_prop_cycle("color", custom_colors)
        subplot_axes = df.plot(subplots=True, ax=axes, sharex=True, legend=False, title=title, style=style)
        if title:
            axes.set_title(title)
        # Iterate through subplot axes and met variables and label the y-axis on each subplot
        for ax, var in zip(subplot_axes, df.columns):
            ax.set_ylabel(var)
    else:
        fig, axes = plt.subplots(figsize=figsize)
        df.plot(ax=axes, title=title, ylabel=ylabel, color=line_color, style=style)

    if legend_list:
        axes.legend(legend_list)

    fig.tight_layout()  # This resolves a lot of layout issues
