"""
@author : Shreyash Garg
Created : 14 Juli 2023

Notes:
Python script for detecting neuronal avalanches
Currently this is supposed to be the main file
Currently the spike plots are saved in pre existing folders for masks


Output : image file saved in respective folder

Comments : Running slower than expected. Current guess: plotting is bad
The same script in ipython produces correct graphs but not when ran through terminal. Needs fixing.
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

#bin multipliers
binsize = [1,2,3,4]

# binsize for PDF
pdf_bin = 10

#initialise fig and axes for plotting
fig, axs = plt.subplots(2, 2);
fig.set_size_inches(10,10);

def pdf_plotter(fig, ax, b, x_data, y_data):
    """
    Function to plot log PDFs for size and duration
    Input arguments:
        fig : fig handle,
        ax : axis handle,
        b : bin multiplier
        x axis : size/time bins and
        y axis : normalised probabilities

        Comments : Very unoptimised currently.
    """

    ax.plot(x_data, y_data, 'bo', markersize=5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    # ax.set_xlim([1,1000])
    ax.set_ylim([0.0001, 1])
    ax.set_xlabel('Size (#neurons)')
    ax.set_ylabel('P(size)')
    fig.suptitle('Neuronal Avalanche (Hilus itbs data). histogram binsize = 10)')
    ax.set_title('binsize = ' + str(b))


# load the result file
filename = 'Hilus_itbs_data_res'
with open(filename, 'rb') as f:
    data_dict = pickle.load(f)


# iterate over nested dict to extract timeseries activity and also total time and time resolution
for prep_date in data_dict.keys():
    # we need to exclude data from last session as time resolution is extremely bad
    if prep_date == '15-03-2022':
        continue
    else:
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
                    print(prep_date, ' ,', group)
                    save_dir = 'G:/HiWi/itbs/code/hilus_masks/'+prep_date+'/'+group + '/'
                    spike_raster_plotter.raster_plotter(spike_data, save_dir, culture, time_resolution)

                """
                binning when data is from calcium imaging
                We use smaller bin multipliers as data has low temporal resolution
                """

                # data_binwise = {}



                #iterate over each binsize and also its corresponding axis handle
                for b,ax in zip(binsize,axs.ravel()):

                    #this can be used later to move frames backward or forward
                    start = 0
                    #make slice of required data
                    temp_array = spike_data[:,start:int(spike_data.shape[1]-(spike_data.shape[1]%b))]
                    #reshape to size of bins
                    temp_array = temp_array.reshape(temp_array.shape[0],temp_array.shape[1]//b,b)
                    #now we extract how many times the same neuron was active in the whole time bin
                    temp_array = np.sum(temp_array,axis=2)
                    #print(np.unique(temp_array))
                    #create array which stores whether neuron was active in the bin or not
                    temp_array = np.where(temp_array == 0, 0, 1)
                    #print(np.unique(temp_array))
                    #now we want to extract how many electrodes were active in each time bin
                    binned_data_bin_i = np.sum(temp_array,axis = 0)



                    """
                    if one needs to check validity of above code then
                    following example can be used

                    b=2
                    x = np.random.randint(20,size = (4,8))
                    print(x)
                    temp_array = x.reshape(x.shape[0],x.shape[1]//b,b)
                    temp_array = np.sum(temp_array,axis=2)
                    print(temp_array)
                    temp_array = np.sum(temp_array,axis = 0)
                    print(temp_array)
                    """

                    """
                    finding avalaches and their properties
                    """
                    #recursive function to calculate avalache length
                    def avalanche_summary(ind,aval_len):
                        if (ind>len(activity) - 2):
                            return ind, aval_len
                        else:

                            if activity[ind + 1] - activity[ind] > 1:
                                return ind + 1,aval_len
                            else:
                                return avalanche_summary(ind + 1,aval_len+1)


                    # lists to store avalanche details
                    aval_len_bin_i = []
                    aval_size_bin_i = []

                    # find all indexes where activity is present
                    activity = np.where(binned_data_bin_i!=0)[0]

                    #find avalanche length and size
                    ind = 0
                    while ind < (len(activity)-2):
                        aval_len = 1
                        end_ind, aval_len = avalanche_summary(ind, aval_len)
                        aval_size = np.sum(binned_data_bin_i[activity[ind]:activity[end_ind]])
                        aval_len_bin_i.append(aval_len)
                        aval_size_bin_i.append(aval_size)
                        #print(activity[ind],aval_len,aval_size)
                        ind = end_ind

                    """
                    Plot size distribution for all bin sizes
                    2x2 plot as we have 4 binsizes
                    """
                    #define bin size for histogram

                    hist_bins = np.arange(0,10000,pdf_bin)
                    prob, _= np.histogram(aval_size_bin_i,bins = hist_bins, density = True, weights = None);
                    #print(counts)
                    pdf_plotter(fig,ax,b,hist_bins[:-1],prob)

plt.savefig('G:/HiWi/itbs/code/'+ 'Avalanche_pdf', bbox_inches='tight', pad_inches=0)