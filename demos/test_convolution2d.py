import numpy as np
from scipy.signal import convolve2d

# Define a 2D array (e.g., an image or a signal matrix)
array_2d = np.array([[1, 2, 3],
					 [4, 5, 6],
					 [7, 8, 9]])

# Define a 2D convolution kernel (e.g., a filter)
kernel = np.array([[1, 0, -1],
				   [1, 0, -1],
				   [1, 0, -1]])

# Perform the 2D convolution
convolved_array = convolve2d(array_2d, kernel, mode='same')

print("Original Array:\n", array_2d)
print("\nKernel:\n", kernel)
print("\nConvolved Array:\n", convolved_array)
