# -*- coding: utf-8 -*-
"""
Script to generate simple histogram from CSV file

Map file to path and change bin number depending on desired resolution of histogram
"""

import pandas as pd
import matplotlib.pyplot as plt

# path of the csv file that needs to be read
# csv file of two columns, assuming the first column is "x" and the second column is "y"
path = r'C:\Users\AD-AnestNorrislab\Downloads\RTPP 5Hz Cre CKOR-cre C84 F2Cre CX_Y_Position_2020_10_16_(9, 49)_.CSV'

df = pd.read_csv(path, names=['x', 'y'])
# the number of bins can be changed to reflect the video resolution
plt.hist2d(df['x'], df['y'], bins=50)
plt.title("2D histogram from CSV file")
plt.show()
