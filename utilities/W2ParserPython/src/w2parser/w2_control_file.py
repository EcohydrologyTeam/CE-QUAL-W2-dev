import os
from datetime import datetime
import shutil

class W2ControlFile:

	def __init__(self, infile: str):
		self.w2_control_infile = infile
		self.w2_control_filepath = os.path.abspath(infile)
		self.w2_control_filename = os.path.basename(self.w2_control_filepath)
		self.base_folder = os.path.dirname(self.w2_control_filepath)
		self.graph_filepath = os.path.join(self.base_folder, 'graph.npt')
		self.load()

	'''
	Load the CE-QUAL-W2 control file
	'''
	def load(self):
		with open(self.w2_control_filepath, 'r') as f:
			self.lines = f.readlines()

		for i, line in enumerate(self.lines):
			# Check if any of the lines contain tabs. These cannot
			# be automatically handled reliably, so throw an error
			# if any tabs are found
			if '\t' in line:
				raise(IOError, f'Error: line {(i + 1)} in {self.w2_control_inpath} contains tabs, which are not supported. Replace all tabs with spaces.')
		
			# Handle card names that vary
			# TSR Layer / TSR Depth card
			if line.upper().startswith('TSR') and 'ETSR' in line.upper():
				self.lines[i] = 'TSR LAYE' + line[8:]

	'''
	Save the revised CE-QUAL-W2 control file
	'''
	def save(self, out_filepath: str = None, save_backup: bool = True):
		if out_filepath == None:
			out_filepath = self.w2_control_filepath
		basename = os.path.basename(os.path.abspath(out_filepath))
		out_filepath = os.path.join(self.base_folder, basename)

		if save_backup:
			self.save_backup_file()

		with open(out_filepath, 'w') as f:
			print('out_filepath: ', out_filepath)
			f.writelines(self.lines)

	'''
	Save a backup of the original control file
	'''
	def save_backup_file(self, outpath: str = None):
		if outpath is None:
			outpath = self.w2_control_filepath + '.bak.' + datetime.now().strftime('%Y_%m_%d_%H%M_%s')
		shutil.copyfile(self.w2_control_filepath, outpath)