import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def ffts(x): 
	''' Performs shifted and normalized fft '''
	return np.fft.fftshift(np.fft.fft(x))/np.sqrt(len(x))

def affts(x): 
	''' Performs shifted and normalized fft along with absolute value '''
	return np.abs(ffts(x))
	
def log2lin(x): 
	'''Returns the inverse logarithm'''
	return 10**(x/10)

def init_figs():
	plt.close('all')
	############## Options to generate nice figures
	fig_width_pt = float(640.0)  # Get this from LaTeX using \showthe\column-width
	inches_per_pt = 1.0 / 72.27  # Convert pt to inch
	golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
	#golden_mean = float(1.0)
	fig_width = fig_width_pt * inches_per_pt  # width in inches
	fig_height = fig_width * golden_mean  # height in inches
	fig_size = [fig_width, fig_height]
	
	params_ieee = {
		'axes.labelsize': 16,
		'font.size': 16,
		'legend.fontsize': 16,
		'xtick.labelsize': 14,
		'ytick.labelsize': 14,
		'text.usetex': True,
		# 'text.latex.preamble': '\\usepackage{sfmath}',
		'font.family': 'serif',
		'font.serif': 'Times New Roman',
		'figure.figsize': fig_size,
		'axes.grid': True
	}
	
	############## Choose parameters you like
	matplotlib.rcParams.update(params_ieee)





