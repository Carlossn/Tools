import pandas as pd
import numpy as np
import itertools
import numpy as np
from pylab import plt, mpl

##############################EXCEL & AUXILIARY FUNCTIONS###########################################################
#### EXCEL-LIKE FUNCTIONS:
# match:Returns list of tuples (row,col) with the dataframe coordinates of the value sought
# offset:Returns a dataframe range or value that is a given number of rows and cols from a coord reference tuple
#### AUXILIAR FUNCTIONS:
# df_filter_var_type: Feature filtering using type() function based on user choice i.e. True(Numeric) or False (String)
# df_filter_rc_by_string: Returns a dataframe with only the rows(cols) and index(col_names) that contain specific strings 
# create_plot: Generate plot for multiple series X and Y
summary = pd.DataFrame({'function': ['match','offset','df_filter_var_type', 'df_filter_rc_by_string','create_plot'],
                        'DES': ['Returns list of tuples (row,col) with the dataframe coordinates of the value sought',
                                'Returns a dataframe range or value that is a given number of rows and cols from a coord reference tuple',
                                'Feature filtering using type() function based on user choice i.e. True(Numeric) or False (String)',
                                'Returns a dataframe with only the rows(cols) and index(col_names) that contain specific strings',
                                'Generate plot for multiple series X and Y']})
summary = summary[summary.columns[::-1]]

############### EXCEL-LIKE FUNCTIONS:
def match(value, dataframe):
    '''
    Returns list of tuples (row,col) with the dataframe coordinates of 
    the value sought:
        list[(x1,y1),(x2,y2),...]
    '''
    bool_df = dataframe.applymap(lambda x: x == value)
    row = bool_df.loc[bool_df.any(1)].index
    col = bool_df.loc[:, bool_df.any(0)].columns
    return list(itertools.product(row, col))


def offset(coord_tuple, dataframe, r_move, c_move, r_size, c_size):
    '''
    Returns a dataframe range or value that is a given number of rows and cols 
    from a coord reference tuple:
        coord_tuple = (x,y) with x and y being row and col number, respectively.
        dataframe = object to be screened
        r_move = number of rows to move from coord_tuple row ref
        c_move = number of cols to move from coord_tuple col ref
        r_size = desired row size for the returned object
        c_size = desired col size for the returned object
    '''
    coord_n = (coord_tuple[0] + r_move, coord_tuple[1] + c_move)
    df_n = dataframe.iloc[coord_n[0]:(coord_n[0] + r_size), coord_n[1]:(coord_n[0] + r_size)]
    return df_n

############## AUX Functions: Transform/filter data:
def df_filter_var_type(dataframe, numeric=True):
    '''
    Feature filtering using type() function based on user choice i.e. 'True'(Numeric) or 'False' (String)
    '''
    if numeric==True:
        type_list = map(lambda x:type(x),dataframe.iloc[0,:])
        type_idx= [i!=str for i in type_list]
        df = dataframe.iloc[:,type_idx]
    else:
        type_list = map(lambda x:type(x),dataframe.iloc[0,:])
        type_idx= [i==str for i in type_list]
        df = dataframe.iloc[:,type_idx]
    return df

##########################################################
def df_filter_rc_by_string(dataframe,str_r,str_c):
    '''
    Returns a dataframe with only the rows(cols) and index(col_names) that contain specific strings 
    Parameters
    ----------
    str_r = string to be sought in rows
    str_c = string to be sought in cols
    '''
    return dataframe.filter(regex = str_c).filter(like=str_r, axis=0)

############################################################
def create_plot(x, y, styles, labels, axlabels):
    '''
    Generate plot for multiple series X and Y
    
    Parameters
    ---------
    x = List format. List of Time Series Arrays representing x-axis
    y = List format. List of Time Series Arrays representing y-axis
    labels = List format. e.g. ['b','b'] if we have two series x and y
    axlabels = List format. Define x-axis and y-axis name to be displayed. 
    '''
    plt.figure(figsize=(10, 6))
    for i in range(len(x)):
        plt.plot(x[i], y[i], styles[i], label=labels[i])
        plt.xlabel(axlabels[0])
        plt.ylabel(axlabels[1])
    plt.legend(loc=0)