"""
28.04.2023
Created by shreyash garg


Python script to extract time series data from confocal microscopy based imaging
The algorithm identifies neurons using morphological openings and closing and having
size above a given pixel area/threshold

This python file works for batch processing multiple files.

This code was written to identify neurons in dentate gyrus in hippocampus
The raw files are tiff files obtained from cropping in imagej

Input: Raw data for all experiments

Output: python dictionary containing time series data of identified neurons
"""


"""
Import libraries
"""
import os
import czifile
import numpy as np
from lxml import etree
import pickle
import cv2 as cv2
from skimage.measure import label
from skimage import io
import pandas as pd
import shutil


"""
Import user defined modules
"""
import masks_visualiser as mv

def get_firing_rate(intensity):
    data = pd.DataFrame({"RawIntDen": intensity})
    rolling_window = 50
    threshold_ratio = 2.5
    data["RawIntDen_median"] = data["RawIntDen"].rolling(rolling_window).median()
    data["RawIntDen_trend"] = data["RawIntDen_median"].fillna(
        data.RawIntDen_median.iloc[rolling_window-1]
    )
    data["RawIntDen_detrend"] = data.RawIntDen - data.RawIntDen_trend
    data["reference"] = (
        data.RawIntDen_detrend.mean() + threshold_ratio * data.RawIntDen_detrend.std()
    )
    data["is_above_threshold"] = (data.RawIntDen_detrend >= data.reference).astype(int)
    data = data.reset_index().rename(columns={"index": "time_order"})
    data["is_above_threshold_change"] = data.is_above_threshold.diff()
    spike_boundary_index = data.loc[
        data.is_above_threshold_change != 0
    ].dropna().time_order
    data["time"] = data.time_order/len(data)*120
    firing_rate = len(spike_boundary_index) / 2 / 120
    data['firing_rate'] = firing_rate
    return data

# specify the path containing all the data
mainpath = 'G:/HiWi/itbs/data'

# important parameters

# specify kernel size for noise removal
noise_kern_size = 5

# specify kernel size for first opening
kern_open1_size = 8

# specify kernel size for second opening
kern_open2_size = 1

# initialise dictionary holding the results
results = {}

# generate subdirectories to store generated mask images
# rewrite if previous results already exists
if os.path.exists('dg_masks'):
    shutil.rmtree('dg_masks')
os.makedirs('dg_masks')

# iterate over all prep dates
for prep_date in next(os.walk('G:/HiWi/itbs/data'))[1]:
    results[prep_date] = {}
    current_dir = mainpath + '/' + prep_date

    os.makedirs('dg_masks/'+prep_date)

    # iterate over all recording time stamps
    for time in next(os.walk(current_dir))[1]:
        results[prep_date][time] = {}
        print(time)
        foldername = current_dir + '/' + time
        print(foldername)

        os.makedirs('dg_masks/'+prep_date+'/'+time)

        # iterate over all cultures one by one
        for file in os.listdir(foldername):
            filename = os.fsdecode(file)

            if (filename.startswith("Video") or filename.startswith("video")) and not (
            filename.endswith('hilus')) and not (filename.endswith('dg')):

                try:

                    # extract data from the cropped data stored in tiff file
                    raw_dg = io.imread(foldername + '/' + filename[:-4] + '_dg.tif')
                except OSError:
                    print("Could not open/read file:")
                    continue

                # extract info from the metadata of the original file

                results[prep_date][time][filename] = {}

                czi = czifile.CziFile(foldername + '/' + filename)

                print('filename:', filename)

                czi_xml_str = czi.metadata()

                # create element tree of metadata
                czi_parsed = etree.fromstring(czi_xml_str)

                # date and time of imaging
                creation_date = (czi_parsed.xpath("//CreationDate")[0]).text

                # size of image (in pixels)
                x_dim = int((czi_parsed.xpath("//SizeX")[0]).text)
                y_dim = int((czi_parsed.xpath("//SizeY")[0]).text)

                # physical dimension of pixel (in meters)
                x_resolution = float((czi_parsed.xpath('//Items//Value')[0]).text)
                y_resolution = float((czi_parsed.xpath('//Items//Value')[1]).text)

                # name of the channel i.e., tdtomato, egfp
                channel0_name = (czi_parsed.xpath('//Channels//Name')[0]).text
                channel1_name = (czi_parsed.xpath('//Channels//Name')[2]).text

                # time resolution of imaging
                time_res = float((czi_parsed.xpath('//LaserScanInfo//FrameTime')[0]).text)

                # total number of frames in the recording
                frames = int((czi_parsed.xpath('//SizeT')[0]).text)

                # overall time of recording
                total_time = frames * time_res

                """
                Store the meta data which will be useful for further processing
                """
                results[prep_date][time][filename]['recording_metadata'] = {}

                results[prep_date][time][filename]['recording_metadata']['x_dim'] = x_dim
                results[prep_date][time][filename]['recording_metadata']['y_dim'] = y_dim

                results[prep_date][time][filename]['recording_metadata']['x_resolution'] = x_resolution
                results[prep_date][time][filename]['recording_metadata']['y_resolution'] = y_resolution

                results[prep_date][time][filename]['recording_metadata']['channel0_name'] = channel0_name
                results[prep_date][time][filename]['recording_metadata']['channel1_name'] = channel1_name

                results[prep_date][time][filename]['recording_metadata']['time_res'] = time_res

                results[prep_date][time][filename]['recording_metadata']['frames'] = frames

                results[prep_date][time][filename]['recording_metadata']['total_time'] = total_time

                """
                data pre processing
                """

                # extracting data for both channels and store separately
                channel0 = raw_dg[:, 0, :, :]
                channel1 = raw_dg[:, 1, :, :]

                """
                neuron contour identification from tdtomato recording
                """

                # select how long to average the recording. first select the type
                # if averaging = full, then average over whole recording
                # if averaging = user_defined then average for first few defined seconds

                averaging = 'full'

                # averaging = 'user_defined'
                # averaging_time = 30 # in seconds

                # create a slice of first few frames which would be used for cell contour detection
                # channel1 is tdtomato

                if averaging == 'full':
                    tomato_slice = channel1[:frames, :, :]
                elif averaging == 'user_defined':
                    frames_averaged = int(averaging_time / time_res)
                    tomato_slice = channel1[:frames_averaged, :, :]

                # average over the slice
                tomato_slice_mean = np.mean(tomato_slice, 0)

                """
                Image processing
                """
                # convert it to float 32 otherwise medianblur would give error
                tomato_slice_mean = np.float32(tomato_slice_mean)

                # remove noise using medianblur
                tomato_slice_lownoise = cv2.medianBlur(tomato_slice_mean, noise_kern_size)

                # perform morphological opening
                # initialise kernel for first opening
                kernel_1 = np.ones((kern_open1_size, kern_open1_size), np.uint8)

                # perform opening
                tomato_slice_lownoise_open1 = cv2.morphologyEx(tomato_slice_lownoise, cv2.MORPH_OPEN, kernel_1)

                # first remove the background from averaged image
                background_removed = tomato_slice_mean - tomato_slice_lownoise_open1

                # initialise kernel for second opening and then open
                kernel_2 = np.ones((kern_open2_size,kern_open2_size), np.uint8)

                # perform second opening
                opened_image = cv2.morphologyEx(background_removed, cv2.MORPH_OPEN, kernel_2)

                # """
                # The boundaries are still not going away so we try to implement erosion
                # """
                #
                # #perform erosion
                # #opened_image = cv2.erode(opened_image,kernel_2,iterations = 1)
                #

                # binarise image using thresholding. The threshold value was determined by looking at raw recording in image viewer
                threshold_val = 100
                max_val = 1

                ret, tomato_cleaned = cv2.threshold(opened_image, threshold_val, max_val, cv2.THRESH_BINARY)

                """
                Neuron identification by identifying connected regions in binarised image
                """

                # find the label for each connected region
                label_im = label(tomato_cleaned)

                # extract all the labels for identified regions
                label_unique = np.unique(label_im)

                # create a dictionary to store all identified regions
                neuron_dict = {}
                for ind in label_unique:
                    # exclude index 0 as it is just the background
                    if ind != 0:
                        # get coordinates of all the pixels belonging to given component
                        neuron_dict[ind] = np.where(label_im == ind)

                # temporary array to visualise contour of all neurons having size about a given threshold
                mask = np.zeros(tomato_cleaned.shape)

                # create separate dictionary which contains data only for putative neurons
                # contains neuronwise data for only those who have size bigger than 20
                neuron_dict_filtered = {}
                for key in neuron_dict.keys():
                    if np.size(neuron_dict[key]) > 20:
                        neuron_dict_filtered[key] = neuron_dict[key]
                        # make all pixels 1 if we consider given component as putative neuron
                        mask[neuron_dict[key]] = 1

                # call masks_plotter function to generate neuron contours image file
                reference_image = channel1[0, :, :]
                save_dir = 'dg_masks/'+prep_date+'/'+time + '/'
                mv.masks_plotter(tomato_cleaned, mask, neuron_dict_filtered, reference_image, save_dir, filename)

                results[prep_date][time][filename]['mask'] = mask
                """
                This section is used to generate data which can be used with masks_visualiser
                to generate neuron mask images with neuron number 
                """

                mask_color = np.zeros((tomato_cleaned.shape[0], tomato_cleaned.shape[1], 3), dtype='int32')
                for r in range(tomato_cleaned.shape[0]):
                    for c in range(tomato_cleaned.shape[1]):
                        if mask[r, c] == 1:
                            mask_color[r, c, 0] = 255

                results[prep_date][time][filename]['mask_color'] = mask_color
                results[prep_date][time][filename]['neuron_dictionary_filtered'] = neuron_dict_filtered

                """
                 alternate way to store timeseries would be to store in a dictionary and pickle dump.
                 I thought of using numpy array but then the data is not too big
                and its easier to keep track of neuron ID in dictionary
                """
                comp_timeseries_dict = {}
                firing_rate_dict = {}

                for neuron in neuron_dict_filtered.keys():
                    # save timeseries data averaged over all pixels
                    comp_timeseries_dict[neuron] = np.mean(
                        channel0[:, neuron_dict_filtered[neuron][0], neuron_dict_filtered[neuron][1]], axis=1)
                    firing_rate_dict[neuron] = get_firing_rate(comp_timeseries_dict[neuron])

                results[prep_date][time][filename]['time_series'] = comp_timeseries_dict
                results[prep_date][time][filename]['firing_rate'] = firing_rate_dict


res_name = 'DG_itbs_data_res'
with open(res_name, 'wb') as f:
    pickle.dump(results, f)