from __future__ import division

import numpy as np

def quantize_ydata(ydata, data):
    """
     To quantize, first we need to determine the quantization level 
     or granularity. Granualarity will be the least common denominator of changes 
    """
    # we want to find the smallest change that has occured in ydata
    u = np.unique(np.abs(ydata))
    min_ydata = np.partition(u, 1)[1]
    if min_ydata == 0:
        min_ydata = 1
    ydata = ydata/min_ydata

    # if ydata is positive, ceil it to convert to integer
    ydata[np.where(ydata >= 0)[0]] = np.ceil(ydata[np.where(ydata >= 0)[0]])

    # if ydata is negative, floor it to convert to integer
    ydata[np.where(ydata < 0)[0]] = np.floor(ydata[np.where(ydata < 0)[0]])

    # When there is a relative decrease, the level is negative,
    # we want to bring all levels >= 0, the minimum level goes to 0 and 
    # the rest accordingly
    min_level = np.min(ydata) 
    ydata += np.abs(min_level)
    return ydata

def time_series_data(data, config):
    """
     pre-process time series data
     data format: each column denotes bikes of
     a particular station, each row belongs to one time stamp
     For example data[i,j] would be number of bikes at station j
     on i th time-stamp
    """
    # we are only treating config.data_columns as X-data
    Xdata = data[:,config.data_columns]

    # ydata is config.time_delta timesteps ahead of xdata
    Xdata = Xdata[:-config.time_delta,:]
    
    # we are only treating config.labels as Y-data
    ydata = data[config.time_delta:,config.label_columns]
    
    # we are interested in the relative change of ydata from Xdata
    # that's why we take a difference and normalize the difference
    ydata_max = np.max(ydata)
    ydata = ((ydata - Xdata[:,:len(config.label_columns)]) / ydata_max) * 100
    # flatten the ydata, need to re evaluate this step when working 
    # with len(config.label_columns) > 1
    ydata = ydata.flatten()

    # the relative changes is a Real value, we want it to be in 
    # discrete steps, that's why quantize the ydata
    ydata = quantize_ydata(ydata, data)

    l = len(Xdata)
    #each row has config.num_steps columns
    Xdata = Xdata[:l - (l%config.num_steps),:]
    ydata = ydata[:l - (l%config.num_steps)]
    Xdata = Xdata.reshape((-1, config.num_steps, config.nr_features))
    ydata = ydata.reshape((-1, config.num_steps))
        
    return Xdata, ydata

def xy_data(data, config):
    """
    pre-process typical X,y data. 
    data format: last column is labels, rest of the columns are features
    """
    Xdata = data[:,:-1]
    ydata = data[:,-1] #last column is labels/ydata

    return Xdata, ydata
