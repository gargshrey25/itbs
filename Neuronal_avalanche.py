"""
@author : Shreyash Garg
Created : 14 Juli 2023

Notes:
Python script for detecting neuronal avalanches
Currently this is supposed to be the main file

Currently the spike plots are saved in pre existing folders for masks

Input :


Output : image file saved in respective folder
"""

"""
Import libraries
"""
import os
import numpy as np
import pickle
import pandas as pd
import scipy as sp
from scipy import signal
import matplotlib.pyplot as plt
import schemdraw
from schemdraw import flow
from scipy.signal import find_peaks


"""
Import user defined modules
"""
import spike_raster_plotter

"""
initialise variables of interest
"""
# multiplier of standard deviation for spike thresholding
thresh_std_scale = 2.5


# load the result file
filename = 'Hilus_itbs_data_res'
with open(filename, 'rb') as f:
    data_dict = pickle.load(f)


# iterate over nested dict to extract timeseries activity and also total time and time resolution
for prep_date in data_dict.keys():
    for group in data_dict[prep_date].keys():
        for culture in data_dict[prep_date][group].keys():
            # print(prep_date,group,culture)
            timeseries_all_neurons = data_dict[prep_date][group][culture]['firing_rate']

            spike_data_all_neurons = {}
            total_time = data_dict[prep_date][group][culture]['recording_metadata']['total_time']
            time_resolution = data_dict[prep_date][group][culture]['recording_metadata']['time_res']

            # print(time_resolution)
            # print(total_time)

            # the identity of neurons dont matter now so store data in numpy array
            # initialise array to store spike data
            spike_data = np.zeros((len(timeseries_all_neurons.keys()),
                                   data_dict[prep_date][group][culture]['recording_metadata']['frames']),
                                  dtype=np.uint8)
            for ind, neuron in enumerate(timeseries_all_neurons.keys()):
                # extract timeseries of nth neuron
                timeseries_neuron_i = timeseries_all_neurons[neuron]["RawIntDen_detrend"].to_numpy()

                # find spikes based on threshold and store them
                threshold = np.median(timeseries_neuron_i) + thresh_std_scale * np.std(timeseries_neuron_i)
                peaks, _ = find_peaks(timeseries_neuron_i, height=threshold)
                spike_data[ind, peaks] = 1

            # plot and save spike visualisation for each culture
            if(len(timeseries_all_neurons)>0):
                print(prep_date,' ,',group)
                save_dir = 'G:/HiWi/itbs/code/hilus_masks/'+prep_date+'/'+group + '/'
                spike_raster_plotter.raster_plotter(spike_data,save_dir,culture,time_resolution)






