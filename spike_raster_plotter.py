
"""
@author : Shreyash Garg
Created : 14 Juli 2023

Notes:
Python script used to visualise spike raster plot for each cell culture
This could have been just a function but I want to increase the functionality later

Input :
        numpy array containing spike data for a given culture
        time resolution of spike data
        save directory
        culture name/filename

Output : image file saved in respective folder
"""

import matplotlib.pyplot as plt

def raster_plotter(spike_data, save_dir, culture,time_resolution):
    plt.imshow(spike_data, aspect='auto', interpolation='none')
    plt.gray()
    plt.ylabel('Neuron ID')
    plt.xlabel('Timestep (Each step is frame time i.e. ' + str(time_resolution)[:5] + ' s)')
    plt.savefig(save_dir+culture + '_raster.png', bbox_inches='tight', pad_inches=0)
