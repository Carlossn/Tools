import pandas as pd
import numpy as np
import sklearn.feature_selection as fs
from sklearn import preprocessing
################################# FEATURE SELECTION TOOLS##########################################################
# feature_select_low_var: feature selection using low variance method to the normalized applied features in dataframe.
summary = pd.DataFrame({'function': ['feature_select_low_var'],
                        'DES': ['Feature selection using low variance method to the normalized applied features in dataframe']})
summary = summary[summary.columns[::-1]]

#########################################################################################################################


def feature_select_low_var(dataframe, vars_name=None, threshold_=1, ret_normal=False):
    '''
    Feature selection using low variance method to the normalized applied features in dataframe. 
    Those features with variance below threshold are eliminated.
    Parameters
    ----------
    dataframe = features-only dataframe (response variable excluded)
    vars_name = list/array of strings with each feature name
    threshold = '1' default. Beware data is normalized so 'threshold = 1' is tantamount to 1x std deviations.
    ret_normal = 'False' default. If 'False' dataframe returned is normalized, otherwise the original data is returned.
    '''

    # Dataframe column names automatic obtention unless provided:
    if vars_name == None:
        try:
            names = dataframe.columns
        except:
            names = vars_name
    else:
        names = vars_name
    # taking out non-numeric vars:
    type_list = map(lambda x: type(x), dataframe.iloc[0, :])
    type_idx = [i != str for i in type_list]
    df = dataframe.iloc[:, type_idx]
    # normalizing numeric variables to get rid of scale effect over variance:
    df_t = preprocessing.StandardScaler().fit_transform(df)
    # Low Variance variable selection:
    selection = fs.VarianceThreshold(threshold=threshold_).fit_transform(df_t)
    # Retrieve selected variable names:
    sel_fit = fs.VarianceThreshold(threshold=threshold_).fit(df_t)
    selec_vars = selec_vars = names[sel_fit.get_support()]

    for i in selection[0, :]:
        temp = i == df_t[0, :]
        selec_vars.append(names[temp][0])

    if ret_normal == True:
        df_sel = pd.DataFrame(selection, columns=selec_vars)
        print('Returned variables are normalized')
    else:
        df_sel = pd.DataFrame(dataframe[selec_vars], columns=selec_vars)
        print('Returned variables are original')

    return df_sel
