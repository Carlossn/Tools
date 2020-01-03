import pandas as pd
import numpy as np
import sklearn.feature_selection as fs
from sklearn import preprocessing
import sklearn.grid_search as gs
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

################################# FEATURE SELECTION TOOLS##########################################################
# feature_select_low_var: feature selection using low variance method to the normalized applied features in dataframe.
# feature_select_univariate: Returns features selected or sorted features ranking from feat_df using univariate tests
# feature_select_tree_model_plot_data: returns df ranking features by importance. Use only for tree-based models
# feature_importance: returns dataframe ranking features by importance for any ML model included in clfs dictionary below.
# PCA_data_chart: returns Principal Components percentage of variance (Individual or Cumulative) and PCs series info (n0bs x PCs).
# feature_select_PC_corr: returns features ranked by overall importance with regards their correlation with Principal components.

summary = pd.DataFrame({'function': ['feature_select_low_var', 'feature_select_univariate', 'feature_select_tree_model_plot_data',
                                     'feature_importance', 'PCA_data_chart', 'feature_select_PC_corr'],
                        'DES': ['Feature selection using low variance method to the normalized applied features in dataframe',
                                'Returns features selected or sorted features ranking from feat_df using univariate tests',
                                'returns df ranking features by importance. Use only for tree-based models',
                                'returns dataframe ranking features by importance for any ML model included in clfs dictionary below',
                                'returns Principal Components percentage of variance (Individual or Cumulative) and PCs series info (n0bs x PCs)',
                                'returns features ranked by overall importance with regards their correlation with Principal components']})
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

##########################################################################################


def feature_select_univariate(feat_df, resp_df, problem='classification', cut_off=0.7, percentile=0.4, p_value=0.05, return_ranking=True):
    '''
    Returns features selected or sorted features ranking from feat_df using univariate tests.
    Parameters
    ----------
    feat_df = dataframe with features data and column names
    resp_df = dataframe or array with response date
    problem = 'classification' default. Chi2 will be used for 'classification' problems and F_stat for 'regression'
    cut_off = 0.75 default. Cut-off threshold to select a variable using the average score from all the modes used.
    percentile = 0.4 default. Variables to be selected expressed as percentile.
    p_value = 0.05 default. P-value threshold used to select variables from the 'fpr','fdr' and 'fwe' modes
    return_ranking = True default. If True it returns a feature ranking, otherwise it will return the dataframe 
    with selected features above required cut-off threshold

    modes definition:
    * percentile = select features based on percentile of the highest scores
    * fpr = select features based on a false positive rate test  = Number Non-rejec False Ho / Tot Number True Ho 
    * fdr = select features based on an estimated false discovery rate = Number Reject True Ho / Total Num Rejections
    * fwe = select features based on family-wise error rate = Probability of making at least one type I error (Reject True Ho)

    '''
    import pandas as pd
    import numpy as np
    import sklearn.feature_selection as fs
    m_list = ['percentile', 'fpr', 'fdr', 'fwe']
    p_list = [0.4, 0.05, 0.05, 0.05]
    sel_df = pd.DataFrame(index=feat_df.columns)
    method = ['fs.chi2' if problem == 'classification' else 'fs.f_regression'][0]
    for m, p in zip(m_list, p_list):
        f = fs.GenericUnivariateSelect(eval(method), mode=m, param=p).fit(feat_df, resp_df)
        sel_df[m] = f.get_support()

    sel_df['average'] = sel_df.apply(np.mean, axis=1)
    if return_ranking == True:
        return sel_df.sort_values(by='average', ascending=False)
    else:
        return feat_df[sel_df[sel_df['average'] > cut_off].index]

#########################################################################################################################


def feature_select_tree_model_plot_data(model, feat_labels, return_data=False):
    '''
    Returns dataframe ranking features by Cost Complexity importance ONLY use for tree-based models (CART,RF, XGB, etc)
    Params
    -------
    model = fitted tree-model created using:
        from sklearn import tree
        tree_model = tree.DecisionTreeClassifier()
        grid_search_tree.fit(x_train, y_train)

    feat_labels = array with feature names
    return_data = False default.  If True it will only return dataframe with ranking instead of a bar plot.

    '''
    import pandas as pd
    f_imp = pd.Series(model.feature_importances_)
    try:
        df = pd.DataFrame(f_imp)
        df.index = feat_labels
    except:
        df = pd.DataFrame(f_imp, index=f_imp.index)
    df = df.rename(columns={0: 'Importance'})
    df.sort_values(by='Importance', ascending=False, inplace=True)
    if return_data == True:
        return df
    else:
        df.plot(kind='bar', figsize=(12, 8))


################################################################################
def feature_importance(model, feat_labels, model_name, return_data=True):
    '''
    Returns dataframe ranking features by importance for any ML model included in clfs dictionary below.
    Params
    -------
    model       = fitted model. You can also include GridSearchCV.fit model, yet enter the CV model parameter name in model_name. 
    model_name  = Check clfs dictionary below. If you use GridsearchCV enter "GridSearchCV".
    feat_labels = array with feature names
    return_data = "False" default to return bar chart. If "True" it returns a dataframe with ranking.
    '''

    clfs = {'RandomForestClassifier': 'feature_importances',
            'ExtraTreesClassifier': 'feature_importances',
            'AdaBoostClassifier': 'feature_importances',
            'LinearRegression': 'coef',
            'LogisticRegression': 'coef',
            'svm.SVC': 'coef',
            'GradientBoostingClassifier': 'feature_importances',
            'GaussianNB': None,
            'DecisionTreeClassifier': 'feature_importances',
            'SGDClassifier': 'coef',
            'KNeighborsClassifier': None,
            'linear.SVC': 'coef'}

    if clfs[model_name] == 'feature_importances':
        try:
            list1 = list(model.feature_importances_)
        except:
            list1 = list(model.best_estimator_.feature_importances_)
    elif clfs[model_name] == 'coef':
        list1 = list(model.coef_.tolist())

    df = pd.DataFrame(list1, index=feat_labels, columns=['Importance'])
    df.sort_values(by='Importance', ascending=False, inplace=True)

    if return_data == True:
        return df
    else:
        df.plot(kind='bar', figsize=(12, 8))

########################################################################################


def PCA_data_chart(dataframe, num_PCs=None, cum_var_chart=True, ret_PCs=True):
    '''
    Returns Principal Components percentage of variance (Individual or Cumulative) and PCs series info (n0bs x PCs). 

    '''

    # check dataframe and take string features out:
    try:
        type_list = map(lambda x: type(x), dataframe.iloc[0, :])
    except:
        dataframe = pd.DataFrame(dataframe)
        type_list = map(lambda x: type(x), dataframe.iloc[0, :])

    type_idx = [i != str for i in type_list]
    df = dataframe.iloc[:, type_idx]
    # normalizing numeric variables to get rid of scale effect over variance:
    data = preprocessing.StandardScaler().fit_transform(df)
    # initialize PCA
    pca = PCA()

    try:
        pca.set_params(n_components=num_PCs)
    except:
        pca.set_params(n_components=int(data.shape[1] + 1))
    pca.fit(data)
    # obtain PCs
    data_pca = pca.transform(data)  # nobs vs PCs

    # chart settings:
    if num_PCs == None:
        n_PC = dataframe.shape[1] + 1
    else:
        n_PC = num_PCs + 1

    if cum_chart == None:
        chart = pca.explained_variance_ratio_
    else:
        chart = np.cumsum(pca.explained_variance_ratio_)

    plt.plot(range(1, n_PC), chart)
    plt.scatter(range(1, n_PC), pca.explained_variance_ratio_)
    plt.xlabel('ith components')
    plt.xticks(range(1, n_PC))
    if cum_chart == False:
        plt.ylabel('Percentage of Variance')
    else:
        plt.ylabel('Cumulative Percentage of Variance')
    plt.show()
    print('See below PCs (nobs x PCs)')
    return pd.DataFrame(data_pca, columns=['PC_' + str(x) for x in range(1, n_PC)])

#########################################################################################


def feature_select_PC_corr(dataframe, vars_name=None, num_PCs=None):
    '''
    Returns features ranked by overall importance with regards their correlation with Principal components.
    The function takes the correlation of each feature with every PC and calculates a feature score from the 
    sum of the product of each feature's correlation and the respective PC's explained variance.
    Parameters
    ----------
    dataframe = dataframe with features(cols) and obs (rows)
    vars_name = list of names for the dataframe columns/features
    num_PCs = number of principal components to be used for the scoring
    '''
    # check dataframe and take string features out:
    try:
        type_list = map(lambda x: type(x), dataframe.iloc[0, :])
    except:
        dataframe = pd.DataFrame(dataframe)
        type_list = map(lambda x: type(x), dataframe.iloc[0, :])

    type_idx = [i != str for i in type_list]
    df = dataframe.iloc[:, type_idx]
    # normalizing numeric variables to get rid of scale effect over variance:
    data = preprocessing.StandardScaler().fit_transform(df)
    # initialize PCA
    pca = PCA()
    try:
        pca.set_params(n_components=num_PCs)
    except:
        pca.set_params(n_components=int(data.shape[1] + 1))
    pca.fit(data)
    # obtain PCs
    data_pca = pca.transform(data)  # nobs vs PCs
    # obtain correl matrix features vs PCs
    pc_df = pd.DataFrame(data_pca, columns=['pc_' + str(x) for x in range(1, data_pca.shape[1] + 1)])
    try:
        data_df = pd.DataFrame(data, columns=['var_' + str(x) for x in vars_name])
    except:
        data_df = pd.DataFrame(data, columns=['var_' + str(x) for x in range(1, data.shape[1] + 1)])

    df = pd.concat([pc_df, data_df], axis=1)
    df_corr = df.corr().filter(regex='pc').filter(like='var', axis=0)
    # Scoring:
    df_corr_pct = df_corr.rank(pct=True)
    var_weight = pca.explained_variance_ratio_  # weights based on PC's Var explained importance
    df_var_score = pd.DataFrame(np.sum(df_corr_pct * var_weight, axis=1), columns=['score'])
    print('Number of PCs choosen explain %d percentage of Vol' % (round(sum(pca.explained_variance_ratio_) * 100, 2)))
    return df_var_score.sort_values(by='score', ascending=False)
