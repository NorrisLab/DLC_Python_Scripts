# -*- coding: utf-8 -*-
"""
Path Tracing for Figure
Heat Map Plot

Analysis perfomed using DeepLabCut

@article{Mathisetal2018,
    title={DeepLabCut: markerless pose estimation of user-defined body parts with deep learning},
    author = {Alexander Mathis and Pranav Mamidanna and Kevin M. Cury and Taiga Abe  and Venkatesh N. Murthy and Mackenzie W. Mathis and Matthias Bethge},
    journal={Nature Neuroscience},
    year={2018},
    url={https://www.nature.com/articles/s41593-018-0209-y}}

 @article{NathMathisetal2019,
    title={Using DeepLabCut for 3D markerless pose estimation across species and behaviors},
    author = {Nath*, Tanmay and Mathis*, Alexander and Chen, An Chi and Patel, Amir and Bethge, Matthias and Mathis, Mackenzie W},
    journal={Nature Protocols},
    year={2019},
    url={https://doi.org/10.1038/s41596-019-0176-0}}

Based Code from Contributed by Federico Claudi (https://github.com/FedeClaudi) for DeepLabCut Utilities
"""

# Importing the toolbox (takes several seconds)
import pandas as pd
from pathlib import Path
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# frame rate of camera in those experiments
fps = 60

DLCscorer = 'DLC_resnet50_TopDownAug29shuffle3_40000'
fig, ax = plt.subplots()

def format_graph(graph_title):
    """
    
    Parameters
    ----------
    graph_title : STRING
        Ideally this should be a string with the animal name. This function will format the graph to Arial 12 font with thin
        lines that is consistent with our general figure formatting

    Returns
    -------
    None.
        This function will save the generated plot in a 600dpi image. By default the images are saved as jpgs but this can be
        changed to a variety of different formats.

    """
    font = {'family': 'Arial',
             'size': 12}
    plt.rc('font', **font)
    plt.rc('lines', linewidth = 1)
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    plt.title(graph_title, fontsize=12)
    plt.savefig(graph_title+".jpg", dpi=600)
    plt.show()
     
def heatmap(video, body_part, color, plot_title):
    """
    
    
    Parameters
    ----------
    video : STRING
        The inital part of the video PRIOR to the DLC_resnet part, this serves as a unique identifier when multiple videos are
        used from the same scorer.
    body_part : STRING
        A string that references the body part that you would like to annotate. This **MUST** be one of the body parts that was
        analyzed using DLC. Ex: head, tailbase, etc. Be cognizant of how this was typed in the config.yaml file.
    color : STRING
        A string that references a color. This can be fully written out in a word or the Hex code can be provided.
    plot_title : STRING
        Title prefix for the plots that will be generated. For the Tracing plots it will be the name of the plot followed by the
        words "Path Tracing". In the case of the heatmap this will name of the plots will be followed by "Heatmap".
    
    Returns
    -------
    None.
        2 plots will be generated. One for the heatmap and the other for the path tracing.

    """
    
    # this will generate the whole stem of the video based on the `video` and `DLCscorer` that was specified
    dataname = str(Path(video).stem) + DLCscorer + 'filtered.h5'
    # print(dataname) ##this can be uncommented to print the full dataname to get an idea of what's going on

    # loading output of DLC by reading the H5 file and storing it into `DataFrame`
    # you could also replace `read_hdf` with `read_csv` to read from a CSV file instead. I mainly worked with H5 files since
    # they were smaller and less cumbersome to deal with
    Dataframe = pd.read_hdf(os.path.join(dataname), errors='ignore')
    
    
    # this part of the code plots the path tracing based on the x and y coordinate values found in the `DataFrame` 
    bpt = body_part
    x_y_cord_df = pd.DataFrame()
    x_y_cord_df['x'] = Dataframe[DLCscorer][bpt]['x'].values
    x_y_cord_df['y'] = Dataframe[DLCscorer][bpt]['y'].values
    # print(x_y_cord_df)
    plt.plot(x_y_cord_df['x'], x_y_cord_df['y'], color=color)
    format_graph(plot_title+" Path Tracing")
    plt.show()

    # this part of the code plots a heatmap based on where the `body_part` was located during analysis. You shouldn't need to
    # change anything else in this code _except_ for the `bin` number. Refer to the <simple heatmap>[url] code for more detail 
    # on what changing the `bin` number does
    plt.hist2d(x_y_cord_df['x'], x_y_cord_df['y'], bins=35)
    
    # these two lines will plot a vertical colorbar that is empty, the default
    cb1=plt.colorbar(orientation="vertical")
    cb1.set_ticks([])
    
    # if you want to plot a more detailed looking colorbar, uncomment this portion of the code. You'll end up generating a color
    # bar which looks like the image below
    # insert image
    
    
    # plt.clim(0, fps) #limit for colorbar
    # sfmt=ticker.ScalarFormatter(useMathText=True) #makes the label look more "math like"
    # cb1=plt.colorbar(orientation="vertical",format=sfmt) #vertical orientation with scientific number formatting
    # cb1.set_label(r'$number of frames$') 
    
    format_graph(plot_title+" Heatmap")
    plt.show()
    
                  
if __name__ == '__main__':

    heatmap(video='TopDown_TestMouse_Top Down', body_part='head', color='#7ca338', plot_title='TopDown Test Animal')


