# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 12:05:14 2021

@author: AD-ANESTNorrisLab
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from textwrap import wrap

# to avoid ~color choosing fatigue~, I have a list of colors that I like to iterate through to avoid thinking about
# things when you don't have to!
my_colors = ['#1f77b4', 'black', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#ff7f0e']


# prevent numpy exponential
# notation on print, default False
np.set_printoptions(suppress=True)
# intializing the dataframe with one empty column to copy values into later on in the script
avg_df = pd.DataFrame(data=None, columns=['Time'])

# some things you might consider changing
# in seconds, how long the rolling average duration needs to be
rolling_avg_duration = 4
# how often to take a sample of the video. If the sample int is 50, then the values will be taken
# every 50 frames and placed in that data frame
sample_int = 120
# this value would need to be calculated based on a video. Run the video through bonsai to see how many pixels is
# in 1cm
pixels_per_cm = 1/42 
fps = 120

body_part_to_plot = 'hindpaw'

def vel_det(file, line_color):
    """
    Parameters
    ----------
    file : STRING
        The h5 file that DLC generates after analysis
    line_color : STRING
        what color you want the lines to appear
    Returns
    -------
    None.
        Generates a velocity plot. Using the x and y coordinate from the chosen body part, calculate the displacement
        if there's a zero value then take the value from before. Afterwards, calculate the absolute value and plot
        the results.
        
        A window will be used which is specified above to roll the velocity based on how many seconds you specify.
        If you do NOT want to roll the velocity, just set this value to 1.


    """    
    
    global new_df
  
    # got most of this part from https://github.com/DeepLabCut/DLCutils/blob/master/Demo_loadandanalyzeDLCdata.ipynb
    
    
    # will read the h5 file into a dataframe
    data_df = pd.read_hdf(path_or_buf=file)
    # gets the list of body parts based on the first row of the h5 file
    bodyparts = data_df.columns.get_level_values(1)
    # gets x and y coordinates based on the second row of the h5 file
    coords = data_df.columns.get_level_values(2)
    bodyparts2plot = bodyparts
    # gets the scorer information that is noted on each excel sheet
    scorer = data_df.columns.get_level_values(0)[0]
    # grabs the time by looking at how many values are in the excel sheet
    Time = np.arange(np.size(data_df[scorer][bodyparts2plot[0]]['x'].values))
    # creates column title with underscores instead of spaces
    column_title = bodyparts + "_" + coords
    data_df.columns = column_title

    # creates a new DataFrame with one column that is empty
    calc_df = pd.DataFrame(data=None, columns=[body_part_to_plot+'_x_filt'])
    
    # calculate the time elapsed per frame and append column
    data_df['Time Elapsed'] = Time / fps

    # figure out how large DataFrame is to iterate through
    list_no = np.arange(0, len(calc_df), 1)
    
    # calculate the difference from row under to row before to find x displacement
    # if there is a zero value, use the previous value, otherwise do nothing
    # then calculate absolute value
    
    # create a new dataframe taking values as requent as the integer that is specified
    calc_df[body_part_to_plot+'_x_filt'] = data_df[body_part_to_plot+'_x'].iloc[::sample_int]
    # calculates the difference by subtracting the current value from the previous value
    calc_df['|diff X|'] = calc_df[body_part_to_plot+'_x_filt'].diff(-1)
    
    # iterate through the list that's the length of the dataframe
    for i in list_no:
        # if the difference is 0 that means the point didn't move very much so we'll take the previous value from before
        if calc_df['|diff X|'].iloc[i] == 0: #if difference zero
            calc_df['|diff X|'].iloc[i] = calc_df['|diff X|'].iloc[i-1] #replace with previous value
        else: # in the case of the else condition, the difference is not zero
            pass # so we will continue on with the code
    calc_df['|diff X|'] = calc_df['|diff X|'].abs() # calculate the absolute value of the difference so velocity will always be positive
    
    list_no_2 = np.arange(0, len(calc_df), 1)
    calc_df[body_part_to_plot+'_y_filt'] = data_df[body_part_to_plot+'_y'].iloc[::sample_int]
    calc_df['|diff Y|'] = calc_df[body_part_to_plot+'_y_filt'].diff(-1)
    for i in list_no_2:
        if calc_df['|diff Y|'].iloc[i] == 0:
            calc_df['|diff Y|'].iloc[i] = calc_df['|diff Y|'].iloc[i-1]
        else:
            pass
    calc_df['|diff Y|'] = calc_df['|diff Y|'].abs()


    # calculating the cummulative sum down the column which is useful for calculating distance travelled
    calc_df['sumX'] = calc_df['|diff X|'].cumsum()
    calc_df['sumY'] = calc_df['|diff Y|'].cumsum()


    # squaring delta x and y values since distance requires squaring these values
    calc_df['deltax^2'] = calc_df['|diff X|']**2
    calc_df['deltay^2'] = calc_df['|diff Y|']**2

    # adding deltaX^2 + deltaY^2
    calc_df['deltaSummed'] = (calc_df['deltax^2'] + calc_df['deltay^2'])
    
    # taking square root of deltaX^2 + deltaY^2, this is basically by taking something to the power of 1/2 so
    # x^(1/2)
    calc_df['eucDist'] = calc_df['deltaSummed']**(1/2)
    # to convert from pixels to cm by multiplying the distance travelled over a period of time
    calc_df['eucDist'] = calc_df['eucDist']*1/pixels_per_cm
    # takes the velocity and rolls it over a certain window and calculates the mean
    calc_df['velocity_roll'] = calc_df['eucDist'].rolling(rolling_avg_duration).mean()
    
    # calculates the time elapsed with the sample integer being taken into consideration
    calc_df['Time Elapsed'] = data_df['Time Elapsed'][::sample_int]
    # print(data_df)

    # grabs the name of the animal from the file
    animal = (' '.join(file.split('_')[:3]))

    # plots the rolled velocity over a period of elapsed time
    plt.plot((calc_df['Time Elapsed']/60), calc_df['velocity_roll'], color=line_color, linewidth=0.5, label=animal)
    
    # put everything into the global dataframe to make it easier to print if troubleshooting is needed
    avg_df[file] = calc_df['velocity_roll']

    
    
def vel_det_paws(file, line_color):
    """
    

    Parameters
    ----------
    file : STRING
        The h5 file that DLC generates after analysis. You can either put the h5 into a folder containing this file
    line_color : STRING
        what color you want the lines to appear.

    Returns
    -------
    None.
        Generates a velocity plot. This is similar to the function above except this calculates an average velocity between the left and right 
        paws. Goes through and calculates the distance travelled and absolute value to calculate velocity.

    """

    # read raw h5 file that is generated from DLC
    data_df = pd.read_hdf(path_or_buf=file)
    bodyparts = data_df.columns.get_level_values(1)
    coords = data_df.columns.get_level_values(2)
    bodyparts2plot = bodyparts
    scorer = data_df.columns.get_level_values(0)[0]
    Time = np.arange(np.size(data_df[scorer][bodyparts2plot[0]]['x'].values))
    column_title = bodyparts + "_" + coords
    data_df.columns = column_title
    calc_df = pd.DataFrame(data=None, columns=[body_part_to_plot+'_x_filt'])
    
    data_df['Time Elapsed'] = Time / fps

    calc_df['right_'+body_part_to_plot+'_x_filt'] = data_df['right_'+body_part_to_plot+'_x'].iloc[::sample_int] #grab every 60 frames
    calc_df['|diff X - 1|'] = calc_df['right_'+body_part_to_plot+'_x_filt'].diff(-1)
    list_no = np.arange(0, len(calc_df), 1)
    for i in list_no:
        if calc_df['|diff X - 1|'].iloc[i] == 0:
            calc_df['|diff X - 1|'].iloc[i] = calc_df['|diff X - 1|'].iloc[i-1]
        else:
            pass
    calc_df['|diff X - 1|'] = calc_df['|diff X - 1|'].abs()

    calc_df['right_'+body_part_to_plot+'_y_filt'] = data_df['right_'+body_part_to_plot+'_y'].iloc[::sample_int] #grab every 60 frames
    calc_df['|diff Y - 1|'] = calc_df['right_'+body_part_to_plot+'_y_filt'].diff(-1)
    for i in list_no:
        if calc_df['|diff Y - 1|'].iloc[i] == 0:
            calc_df['|diff Y - 1|'].iloc[i] = calc_df['|diff Y - 1|'].iloc[i-1]
        else:
            pass  
    calc_df['|diff Y - 1|'] = calc_df['|diff Y - 1|'].abs()

    calc_df['left_'+body_part_to_plot+'_x_filt'] = data_df['left_'+body_part_to_plot+'_x'].iloc[::sample_int] #grab every 60 frames
    calc_df['|diff X - 2|'] = calc_df['left_'+body_part_to_plot+'_x_filt'].diff(-1)
    for i in list_no:
        if calc_df['|diff X - 2|'].iloc[i] == 0:
            calc_df['|diff X - 2|'].iloc[i] = calc_df['|diff X - 2|'].iloc[i-1]
        else:
            pass    
    calc_df['|diff X - 2|'] = calc_df['|diff X - 2|'].abs()

    calc_df['left_'+body_part_to_plot+'_y_filt'] = data_df['left_'+body_part_to_plot+'_y'].iloc[::sample_int] #grab every 60 frames
    calc_df['|diff Y - 2|'] = calc_df['left_'+body_part_to_plot+'_y_filt'].diff(-1)
    for i in list_no:
        if calc_df['|diff Y - 2|'].iloc[i] == 0:
            calc_df['|diff Y - 2|'].iloc[i] = calc_df['|diff Y - 2|'].iloc[i-1]
        else:
            pass     
    calc_df['|diff Y - 2|'] = calc_df['|diff Y - 2|'].abs()


    # calculating the cummulative sum down the column
    calc_df['sumX - 1'] = calc_df['|diff X - 1|'].cumsum()
    calc_df['sumY - 1'] = calc_df['|diff Y - 1|'].cumsum()

    calc_df['sumX - 2'] = calc_df['|diff X - 2|'].cumsum()
    calc_df['sumY - 2'] = calc_df['|diff Y - 2|'].cumsum()

    
    # squaring delta x and y values
    calc_df['deltax^2 - 1'] = calc_df['|diff X - 1|']**2
    calc_df['deltay^2 - 1'] = calc_df['|diff Y - 1|']**2

    calc_df['deltax^2 - 2'] = calc_df['|diff X - 2|']**2
    calc_df['deltay^2 - 2'] = calc_df['|diff Y - 2|']**2

    # adding deltaX^2 + deltaY^2
    calc_df['deltaSummed - 1'] = (calc_df['deltax^2 - 1'] + calc_df['deltay^2 - 1'])
    calc_df['deltaSummed - 2'] = (calc_df['deltax^2 - 2'] + calc_df['deltay^2 - 2'])
    
    # taking square root of deltaX^2 + deltaY^2
    calc_df['eucDist - 1'] = (calc_df['deltaSummed - 1']**(1/2))*pixels_per_cm #stop
    calc_df['velocity_roll - 1'] = calc_df['eucDist - 1'].rolling(rolling_avg_duration).mean()

    calc_df['eucDist - 2'] = (calc_df['deltaSummed - 2']**(1/2))*pixels_per_cm
    calc_df['velocity_roll - 2'] = calc_df['eucDist - 2'].rolling(rolling_avg_duration).mean()
    # calc_df['velocity_roll - 2'] = calc_df['eucDist - 2'].rolling(int(rolling_avg_duration*fps)).mean()


    animal = (' '.join(file.split('_')[:3]))
    calc_df['Time Elapsed'] = data_df['Time Elapsed'][::sample_int]
    calc_df['velocity_roll - 1-A'] = calc_df['velocity_roll - 1']
    calc_df['velocity_roll - 2-A'] = calc_df['velocity_roll - 2']
    plt.plot(calc_df['Time Elapsed']/60, (calc_df['velocity_roll - 1-A'] + calc_df['velocity_roll - 2-A'])/2, color=line_color, linewidth=0.5, label=animal)


    avg_df[file+"-1"] = calc_df['velocity_roll - 1-A']
    avg_df[file+"-2"] = calc_df['velocity_roll - 2-A']
    
    alpha = avg_df.copy()
    print(alpha)
    
def graph_things(title):
    """
    

    Parameters
    ----------
    title : STRING
        Title to be plotted on the graph.

    Returns
    -------
    None.
        Creates a graph and saves the plot if desired.

    """
    # plots a legend
    leg = plt.legend()
    # font attributes, this can be changed to whatever font or font size you'd like
    font = {'family': 'Arial',
            'size': 12}
    plt.rc('font', **font)
    # you can change the linewidth if need be
    plt.rc('lines', linewidth = 1)
    # for the legend, you can change the line width of the legend itself
    for i in leg.legendHandles:
        i.set_linewidth(3)
    plt.legend(loc='upper right')
    plt.xlabel('time (minutes)', fontsize=12)
    plt.ylabel('velocity (cm/second)', fontsize=12)
    plt.title("\n".join(wrap(title)))
    # if you want to save the generated plot, un-comment the next line
    # plt.savefig(title+".png", dpi=600)
    
if __name__ == '__main__':

    
    vel_det_paws(file='Vglut-cre_C269_Day2_M0_side_viewDLC_resnet50_FST-90May6shuffle1_1030000filtered.h5', line_color=my_colors[2])
    graph_things("Velocity of "+body_part_to_plot)
    
    # graph_things("[Forced Swim Velocity Rolling Average "+str(rolling_avg_duration)+" second interval] Vglut-cre C269 M0 - Side View Hindpaws")

    plt.show()