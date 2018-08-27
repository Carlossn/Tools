import pandas as pd
import numpy as np
import itertools

##############################EXCEL & AUXILIARY FUNCTIONS###########################################################
#### EXCEL-LIKE FUNCTIONS:
# match:Returns list of tuples (row,col) with the dataframe coordinates of the value sought
# offset:Returns a dataframe range or value that is a given number of rows and cols from a coord reference tuple
#### AUXILIAR FUNCTIONS:
# df_filter_var_type: Feature filtering using type() function based on user choice i.e. True(Numeric) or False (String)
summary = pd.DataFrame({'function': ['match','offset','df_filter_var_type'],
                        'DES': ['Returns list of tuples (row,col) with the dataframe coordinates of the value sought',
                                'Returns a dataframe range or value that is a given number of rows and cols from a coord reference tuple',
                                'Feature filtering using type() function based on user choice i.e. True(Numeric) or False (String)']})
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