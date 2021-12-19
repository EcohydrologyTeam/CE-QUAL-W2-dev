import pandas as pd
import numpy as np

def get_record(infile, widths, nlines, record_line, record_nlines):
	start = record_line - 1
	end = nlines - (record_line + record_nlines)
	if end < 0:
		end = 0
	data = pd.read_fwf(infile, widths=widths, skiprows=start, skipfooter=end)
	return data

def reshape_wrapped_fields(data: pd.DataFrame, data_type: type):
	field1 = data.iloc[0,0]
	header = data.columns
	values = data.values[:,1:].flatten().tolist()
	card_name = data.columns[0]
	field_name = data.columns[1]
	values2 = [field1] + values
	values2 = [values2] # Create 2D list
	columns = [card_name] + [field_name]*len(values)
	data2 = pd.DataFrame(values2, columns=columns)
	# Make sure the data type of all the columns is correct
	convert_dict = {field_name: data_type}
	data2 = data2.astype(convert_dict)
	data2 = data2.dropna(axis=1)
	# Append the first field back to the data frame if all of the fields were empty and therefore were removed by dropna
	if len(data2.columns) == 1:
		data2[field_name] = np.nan
	return data2