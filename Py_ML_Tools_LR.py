import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from scipy.stats import wilcoxon
from scipy.stats import kruskal
from scipy.stats import friedmanchisquare
import statsmodels as sm
import statsmodels.api as sfm
%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
from statsmodels.graphics.gofplots import ProbPlot
import warnings
import matplotlib.gridspec as gridspec

################################# LR MODEL TOOLS##########################################################
# Functions HELP: import the file and load summary dataframe to check functions available
############
# transf_comparison: Returns key stats, transformed series dataframe and a density plot comparison showing all the transformations
# dist_similarity_test: distribution tests statistics and p-values for a list of variables (input in string format)
# normality_tests: returns key statistics and multiple normality tests p-values for one or more vars
# OLS_Assumption_Tests: check OLS model assumptions: linearity, normality, homoced and non-autorcorrel
# OLS_Assumptions_Plot: plot OLS model assumptions: linearity, normality, homoced and non-autorcorrel
# influence_cook_plot: spot points with outlier residuals (outliers) and high leverage that distort the model
# cook_dist_plot: cook distance measures the effect of deleting a given observation over the regression fit.
# corr_mtx_des: description and correlation matrix to spot features that display multicollinearity
# multivar_LR_plot: plots multiple univar Y vs X regressionsto confirm if there's a relevant univ relationship.
# R_avplot: Similar to R avplot, it compares Y vs X resids (y-axis) against each Xi vs all other xi resids.
# vif_info_clean: get VIF per feature and obtain a clean df without features with VIF>threshold
# dummy_generator: creates new dummy variables and returns a new df with original data plus new dummy variables
# DA_plot_classes: Returns key stats and normality tests as well as probability density functions.

summary = pd.DataFrame({'function': ['transf_comparison','dist_similarity_test', 'normality_tests', 'OLS_Assumption_Tests', 'OLS_Assumptions_Plot', 'influence_cook_plot',
                                     'cook_dist_plot', 'corr_mtx_des', 'multivar_LR_plot', 'R_avplot', 'vif_info_clean',
                                     'dummy_generator', 'DA_plot_classes'],
                        'DES': ['key stats, transformed series dataframe and a density plot comparison showing all the transformations',
                                'distribution tests statistics and p-values for a list of variables (input in string format)',
                                'returns key statistics and multiple normality tests p-values',
                                'check OLS model assumptions: linearity, normality, homoced and non-autorcorrel',
                                'plot OLS model assumptions: linearity, normality, homoced and non-autorcorrel',
                                'spot points with outlier residuals (outliers) and high leverage that distort the model',
                                'cook distance measures the effect of deleting a given observation over the regression fit',
                                'description and correlation matrix to spot features that display multicollinearity',
                                'plots multiple univar Y vs X regressionsto confirm if there is a relevant univ relationship',
                                'Similar to R avplot, it compares Y vs X resids (y-axis) against each Xi vs all other xi resids',
                                'get VIF per feature and obtain a clean df without features with VIF>threshold',
                                'creates new dummy variables and returns a new df with original data plus new dummy variables',
                                'Returns key stats and normality tests as well as probability density functions'
                                ]})
summary = summary[summary.columns[::-1]]

############################################################################################################
def transf_comparison(y, df_only=False):
    '''
    Returns key stats, transformed series dataframe and a density plot comparison showing all the transformations
    Transformations included: Box-Cox and power transformations for 0.5,1,2,3,4,5 powers.
    Parameters:
    y = series/array to be analyzed/transformed
    df_only = 'False' default. If 'True' then it returns key stats and density plot comparison.
    '''
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from scipy import stats
    bc = stats.boxcox(y, lmbda=None, alpha=None)[0]
    y_bc = stats.boxcox(y, lmbda=None, alpha=None)[0]
    lambda_bc = bc[1]
    df = pd.DataFrame({'Original': y,'Box-Cox':y_bc})
    for i in [0.5,1,2,3,4,5]:
        df[str(i)] = np.power(y,i)

    df_n = pd.DataFrame([df.mean(axis=0), df.median(axis=0), df.std(axis=0), df.skew(axis=0), df.kurt(axis=0)+3],
                        columns=['Original', 'Box-Cox','0.5','1','2','3','4','5'],
                        index=['mean','med','std','skew','kurt'])
    if df_only != True:
        print(df_n.round(3))
        print('*'*75)
        print('Transformed Series below:')
        print('note: optimal lambda for Box-Cox is %s'%(lambda_bc))
        for col in df.columns:
                # Draw the density plot
                sns.distplot(df[col], hist = False, kde = True,
                             kde_kws = {'linewidth': 3},
                             label = col)
                plt.xlim(-np.average(df_n.ix['std'])*2,np.average(df_n.ix['std'])*2)
        return df
    else:
        return df
############################################################################################################
def dist_similarity_test(*args):
    '''
    Returns multiple distribution tests statistics and p-values for a list of variables (input in string format)
    Ho Null: samples have the same distribution
    Tests performed are:
     - MW_test = Mann-Whitney U
     - W_test = Wilcoxon
     - K_test = Kruskal
     - F_test = Friedman    '''

    nvar = len(args)
    names = ', '.join(args)

    if nvar > 2:
        MW_test = eval('mannwhitneyu(' + names + ')')
        W_test = [np.nan, np.nan]
        K_test = eval('kruskal(' + names + ')')
        F_test = eval('friedmanchisquare(' + names + ')')
    else:
        MW_test = eval('mannwhitneyu(' + names + ')')
        W_test = eval('wilcoxon(' + names + ')')
        K_test = eval('kruskal(' + names + ')')
        F_test = [np.nan, np.nan]

    df = pd.DataFrame([MW_test, W_test, K_test, F_test],
                     index=['MW_test', 'W_test', 'K_test', 'F_test'])
    print('Ho: Same Distribution')
    return df


##################################################################################################################


def normality_tests(x, show_pv=True):
    '''
    Returns dataframe with different normality tests statistics and p-values
    x = univariate array or multivariable dataframe
    show_pv= True default. If False, it will show only stats.
    '''
    # show pv or statistic:
    opt = [1 if show_pv == True else 0][0]
    # check if x i univariate or multivariate df
    try:
        nvar = len(x.columns)
    except:
        nvar = 1

    x = [pd.DataFrame(x) if nvar == 1 else x][0]

    df = pd.DataFrame()

    for i in range(nvar):
        mean = np.mean(x.iloc[:, i])
        std = np.std(x.iloc[:, i])
        skew = stats.skew(x.iloc[:, i])
        kurt = stats.kurtosis(x.iloc[:, i], fisher=False)
        JB_test = list(stats.jarque_bera(x.iloc[:, i]))
        SW_test = list(stats.shapiro(x.iloc[:, i]))
        DA_test = list(stats.normaltest(x.iloc[:, i]))
        if nvar > 1:
            df[x.columns[i]] = pd.Series([mean, std, skew, kurt,
                                          JB_test[opt], SW_test[opt], DA_test[opt]],
                                         index=['mean', 'std', 'skew', 'kurt',
                                                'JB_test', 'SW_test', 'DA_test'])
        else:
            df['x'] = pd.Series([mean, std, skew, kurt,
                                 JB_test[opt], SW_test[opt], DA_test[opt]],
                                index=['mean', 'std', 'skew', 'kurt',
                                       'JB_test', 'SW_test', 'DA_test'])
    print('Null Hypothesis: Normality - "average" row only museful with pvalues')
    return df.round(2)

#############################################################################################################


def LR_Assumptions_Tests(x, y, type='lr'):
    '''
    Helpful to understand whether or not the model meets the traditional
    OLS model assumptions: linearity, normality, homoskedasticity and non-autocorrelation
    x = explanatory vars/features array/df
    y = dependent var array
    type = 'lr' default for Linear Regression. For Logit (Logistic Regression) enter 'logit'
    '''
    # unvariate or multivar LR:
    try:
        x.columns
    except:
        univariate = True
    else:
        univariate = False

    # fitting model:
    if type == 'lr':
        model_fit = sfm.OLS(y, sfm.add_constant(x)).fit()  # new regression fit as it works with sm.fit() only
    else:
        model_fit = sfm.Logit(y, sfm.add_constant(x)).fit()
    # model residuals
    model_residuals = y - model_fit.predict(sfm.add_constant(x))
    # TESTS:
    # linearity Tests:
    if univariate == True:
        HG_test = list(sm.stats.diagnostic.linear_harvey_collier(model_fit))  # Ho Linearity, Harvey-Collier Test
    else:
        HG_test = [np.NAN, np.NAN]
    # Normality Tests:
    JB_test = list(stats.jarque_bera(model_residuals))  # Ho Normality, Jarque-Bera Test
    # Homocedasticity Tests (Ho: Homocedasticity)
    if univariate == True:
        BP_test = list(sm.stats.diagnostic.het_breuschpagan(model_residuals, x.reshape(-1, 1))[2:4])  # Breusch-Pagan Lagrange Multiplier
    else:
        BP_test = list(sm.stats.diagnostic.het_breuschpagan(model_residuals, x)[2:4])
    W_test = list(sm.stats.diagnostic.het_white(model_residuals, sfm.add_constant(x))[2:4])  # White test
    GQ_test = list(sm.stats.diagnostic.het_goldfeldquandt(model_residuals, sfm.add_constant(x))[0:2])  # Ho diff here: Hetereoced= Var(resides)=Var(X)
    # Non-Autocorrel Tests (Ho: No autocorrelation resids)
    DW_stat = sm.stats.stattools.durbin_watson(model_residuals)
    DW_test = [DW_stat, np.NAN]  # Ho: No Autocorrelation  ,Durbin-Watson
    LJB_output = sm.stats.diagnostic.acorr_ljungbox(model_residuals, lags=int(round(np.log(len(model_residuals)), 0)))  # if Lags=None => default maxlag= ‘min((nobs // 2 - 2), 40)’ # Lags=None => default maxlag= ‘min((nobs // 2 - 2), 40)
    LJB_test = [np.max(LJB_output[0]), np.min(LJB_output[1])]
    BG_test = list(sm.stats.diagnostic.acorr_breusch_godfrey(model_fit, nlags=int(round(np.log(len(model_residuals)), 0)))[2:4])  # Breusch Godfrey Lagrange Multiplier tests
    # Summary DF:
    df = pd.DataFrame([HG_test, JB_test, BP_test, W_test, GQ_test, DW_test, LJB_test, BG_test],
                      columns=['statistic', 'pvalue'],
                      index=['HG Test - Ho: Linearity', 'JB Test - Ho: Normality',
                             'BP Test - Ho: Homoced', 'W Test - Ho: Homoced', 'GQ Test - Ho: Heteroced',
                             'DW Test - Ho: Non-Autocorrel', 'LJB Test - Ho: Non-Autocorrel', 'BG Test - Ho: Non-Autocorrel'])
    return df

############################################################################################################################


def OLS_Assumptions_Plot(x, y, re_type='norm', met_mulcol='mean'):
    '''
    Plotting key charts to check OLS assumptions: linearity, normality, homoskedasticity and non-autocorrelation
    Parameters explanation:
    x = series or dataframe with features
    y = series with response  variabe
    re_type = residual type for heteroced analysis. Normalized resids('norm') as default. Othe options are:
            'standard' = residuals from our model
            'abs_sq_norm' = absolute squared normalized resids  (default)
            'norm' = normalized residuals aka studentized residuals
    met_mulcol = method use to plot intra-corell between features. 'mean' default. The user can enter 'min' or
     'max' also to understand extreme correlation of a specific variable i against all the others.
    '''

    # unvariate or multivar LR:
    try:
        x.columns
    except:
        univariate = True
    else:
        univariate = False

    # Calculations:
    # fitting model:
    model_fit = sfm.OLS(y, sfm.add_constant(x)).fit()  # new regression fit as it works with sm.fit() only
    # calculations required:
    # fitted values (need a constant term for intercept)
    model_fitted_y = model_fit.fittedvalues
    # model residuals
    model_residuals = model_fit.resid
    # normalized residuals
    model_norm_residuals = model_fit.get_influence().resid_studentized_internal
    # absolute squared normalized residuals
    model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))
    # absolute residuals
    model_abs_resid = np.abs(model_residuals)
    # leverage, from statsmodels internals
    model_leverage = model_fit.get_influence().hat_matrix_diag
    # cook's distance, from statsmodels internals
    model_cooks = model_fit.get_influence().cooks_distance[0]

    # Formulas:
    def mcol_corr_plot(dataframe, method=met_mulcol):
        '''
        Plot the mean, max or min (user choice) average correlation of each feature against all the others:
        method = mean default. The user can enter 'min' or 'max' also to understand extreme correlation of a
        specific variable i against all the others.
        '''
        objects = dataframe.columns.values
        y_pos = np.arange(len(objects))
        corr = dataframe.corr()
        if method == 'min':
            corr_ = corr[corr < 1].min().values
        elif method == 'max':
            corr_ = corr[corr < 1].max().values
        else:
            corr_ = corr[corr < 1].mean().values
        plt.bar(y_pos, corr_, align='center', alpha=0.5)
        plt.xticks(y_pos, objects, rotation=90)
        plt.ylabel('Correlation')
        plt.title('Multicollinearity - ' + method + ' per feature ')

    def linearity_plot(x, y):
        plt.scatter(x, y, c='b', lw=1.5, label='Original Data')
        plt.plot(x, model_fitted_y, c='r', lw=1.5, label='Linear Model')
        plt.xlabel('X = Feature(s)')
        plt.ylabel('Y = Dependent variable')
        plt.title('Linearity Check')
        plt.grid(True)
        plt.legend()

    def normality_plot(ts):
        '''
        Check a tseries (i.e.residuals) independence assumption comparing abs.squared resids vs fitted values
        '''
        stats.probplot(ts, plot=plt)
        plt.title('Normality Check')
        plt.grid(True)

    def heteroced_plot(x, y, resid_type=re_type):
        '''
        Check residuals acf from y~x comparing transformed residuals vs fitted values
        reside_type = choose betweeen 3 options:
            'standard' = residuals from our model
            'abs_sq_norm' = absolute squared normalized resids  (default)
            'norm' = normalized residuals aka studentized residuals
        '''
        if resid_type == 'abs_sq_norm':
            plt.scatter(model_fitted_y, model_norm_residuals_abs_sqrt, c='r', lw=1.5)
            plt.axhline(np.mean(model_norm_residuals_abs_sqrt))
            plt.ylabel('Abs.Sqr. Norm Residuals')
        elif resid_type == 'standard':
            plt.scatter(model_fitted_y, model_residuals, c='r', lw=1.5)
            plt.axhline(np.mean(model_residuals))
            plt.ylabel('Residuals')
        else:
            plt.scatter(model_fitted_y, model_norm_residuals, c='r', lw=1.5)
            plt.axhline(np.mean(model_norm_residuals))
            plt.ylabel('Normalized Residuals = Studentized')

        plt.xlabel('Fitted Values')
        plt.title('Heterocedasticity Check')
        plt.grid(True)
        plt.legend()

    def acf_plot(ts, lags_=40):
        ts_ = pd.Series(ts)
        list1 = []
        for i in range(min(len(ts), lags_)):
            list1.append(ts_.autocorr(i))
        df = pd.DataFrame(list1, columns=['ts'])
        df['ts'].plot.bar()
        plt.title('Autocorrel Plot')
        plt.axhline(np.std(list1))
        plt.axhline(-np.std(df.dropna()['ts']), c='g')
        plt.axhline(np.std(df.dropna()['ts']), c='g')
        plt.axhline(-2 * np.std(df.dropna()['ts']), c='r')
        plt.axhline(2 * np.std(df.dropna()['ts']), c='r')

    # Plots:
    plt.style.use('seaborn')  # pretty matplotlib plots
    plt.figure(figsize=(15, 10))
    ax1 = plt.subplot(221)
    if univariate == True:
        linearity_plot(x, y)
    else:
        mcol_corr_plot(x)
    ax2 = plt.subplot(222)
    normality_plot(model_residuals)
    ax3 = plt.subplot(223)
    heteroced_plot(x, y)
    ax4 = plt.subplot(224)
    acf_plot(model_residuals)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=0.5, hspace=0.5)
    plt.show()
    warnings.filterwarnings("ignore", module="matplotlib")


#############################################################################################################

def influence_cook_plot(model_fit, alpha_=0.05, criterion_="cooks"):
    '''
    Data points with large residuals (outliers) and/or high leverage may distort the outcome
    and accuracy of a regression. This chart represent obs leverage(x-axis) vs normalized (studentized)
    residuals (y-axis) with bubble size measuring cook distance:
    - Studentized residuals = normalized residuals from the model
    - Leverage = measures how different an observed value is very different from that predicted by the model.
    - Cook distance = measures the effect of deleting a given observation
    params:
    - model = OLS fitted model
    - alpha = to identify large studentized residuals. Large means abs(resid_studentized) > t.ppf(1-alpha/2,
    dof=results.df_resid)
    - criterion = 'cooks' activates cook distance as bubble size
    '''
    fig, ax = plt.subplots(figsize=(10, 5))
    fig = sm.graphics.influence_plot(model_fit, alpha=alpha_, ax=ax, criterion="cooks")

#############################################################################################################


def cook_dist_plot(model_fit):
    '''
    Cook distance = measures the effect of deleting a given observation
    This plot shows if any outliers have influence over the regression fit.
    Anything outside the group and outside “Cook’s Distance” lines, may have an influential effect on model fit.

    '''

    # Calculations:
    # calculations required:
    # fitted values (need a constant term for intercept)
    model_fitted_y = model_fit.fittedvalues
    # model residuals
    model_residuals = model_fit.resid
    # normalized residuals
    model_norm_residuals = model_fit.get_influence().resid_studentized_internal
    # absolute squared normalized residuals
    model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))
    # absolute residuals
    model_abs_resid = np.abs(model_residuals)
    # leverage, from statsmodels internals
    model_leverage = model_fit.get_influence().hat_matrix_diag
    # cook's distance, from statsmodels internals
    model_cooks = model_fit.get_influence().cooks_distance[0]

    plt.figure(figsize=(10, 5))
    plt.scatter(model_leverage, model_norm_residuals, alpha=0.5)
    sns.regplot(model_leverage, model_norm_residuals,
                scatter=False,
                ci=False,
                lowess=True,
                line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

    plt.xlim(0, 0.20)
    plt.ylim(-3, 5)
    plt.title('Residuals vs Leverage')
    plt.xlabel('Leverage')
    plt.ylabel('Standardized Residuals')

    # annotations
    leverage_top_3 = np.flip(np.argsort(model_cooks), 0)[:3]

    for i in leverage_top_3:
        plt.annotate(i, xy=(model_leverage[i], model_norm_residuals[i]))

    # shenanigans for cook's distance contours
    def graph(formula, x_range, label=None):
        x = x_range
        y = formula(x)
        plt.plot(x, y, label=label, lw=1, ls='--', color='red')

    p = len(model_fit.params)  # number of model parameters

    graph(lambda x: np.sqrt((0.5 * p * (1 - x)) / x),
          np.linspace(0.001, 0.200, 50),
          'Cook\'s distance')  # 0.5 line
    graph(lambda x: np.sqrt((1 * p * (1 - x)) / x),
          np.linspace(0.001, 0.200, 50))  # 1 line
    plt.legend(loc='upper right')


################################# OLS SIMPLE MODEL TOOLS##########################################################
##################################################################################################################

# MULTICOLLINEARITY tools:
#################################################################################################################
def corr_mtx_des(dataframe, method_='pearson', per=1, threshold=0.7):
    '''
    Calculates correlation offering several options:
        * method_= "pearson" default. The methods available are:
            info: http://www.statisticssolutions.com/correlation-pearson-kendall-spearman/
            i) pearson: both vars should meet normality, linearity and homoscedasticity.
            ii) kendall: non-parametric test. Rank correlation measure.
            iii) spearman: non-parametric test. Vars must be of ordinal type and both
            need to be monotonically related to each other.
        * per = default 1. Number of periods to consider for calculation purporses. For instance,
            per =3 will allow to run 3-period/obs rolling correlations.
        * threshold = None default. If a float is given, the heatmap will only highlight correl points above that number.
    '''
    df = dataframe.corr(method=method_, min_periods=per)

    if threshold == None:
        df_n = df
    else:
        df_n = df[df > threshold]

    f, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(df_n, annot=True, mask=np.zeros_like(df, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
                square=True, ax=ax)

    df_des = dataframe.describe()
    return df_des
#################################################################################################################


def multivar_LR_plot(dataframe, y_name, logistic_=False, logx_=False):
    '''
    Plots multiple univariate regressions against Y to understand whether or not there're clear relationships
    before running a multiple LR:

        * dataframe = it contains both response (y) and feature (x) variables.
        * y_name = name of the column for the response variable in the dataframe
        * logistic = boolean type. Default False. True will allow to logistic regression of Y is binary.
        * logx = boolean type. Default False. If True it will transform X using log function before running the model.

    '''

    Y = dataframe[y_name]
    X = dataframe.drop(y_name, 1)

    rows = len(X.columns)
    fig = plt.figure(figsize=(14, 33))
    gs = gridspec.GridSpec(rows, 2)

    for i in range(rows):
        ax1 = plt.subplot(gs[i, 0])
        ax2 = plt.subplot(gs[i, 1])
        sns.regplot(Y, X.iloc[:, i], ax=ax1, logistic=logistic_, logx=logx_)
        ax1.set_title('')
        ax1.set_xlabel('')
        ylim = ax1.get_ylim()
        X[X.columns[i]].hist(bins=50, ax=ax2, orientation='horizontal')
        ax2.set_ylim((ylim[0], ylim[1]))
        ax2.set_xlabel('')
        ax2.set_xlim((0, 200))
        for tick in ax2.yaxis.get_major_ticks():
            tick.label1On = False
            tick.label2On = True
        if i != 0:
            ax1.set_xticklabels([''])
            ax2.set_xticklabels([''])
        else:
            ax1.set_title('Y \n', size=15)
            ax2.set_title('count \n', size=15)
            for tick in ax1.xaxis.get_major_ticks():
                tick.label1On = False
                tick.label2On = True
            for tick in ax2.xaxis.get_major_ticks():
                tick.label1On = False
                tick.label2On = True
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
#########################################################################################


def LR_avplot(dataframe, y_name, logistic_=False, logx_=False):
    '''
    Similar to R avplot, it compares our overall regression resids (y vs all xi, y-axis) against each
    feature residuals (resids from xi vs all other xis). When a clear linear relationship between y vs all xi
    resids and a particular xi vs all other xi resids is discovered, it can be interpreted as unique info coming
    from that that particular xi and therefore that variable has to remain in our ultimate model.

        * dataframe = it contains both response (y) and feature (x) variables.
        * y_name = name of the column for the response variable in the dataframe
        * logistic = boolean type. Default False. True will allow to logistic regression of Y is binary.
        * logx = boolean type. Default False. If True it will transform X using log function before running the model.

    '''

    # Define Y and X:
    Y = dataframe[y_name]
    X = dataframe.drop(y_name, 1)

    cols = len(X.columns)
    fig = plt.figure(figsize=(20, 33))
    gs = gridspec.GridSpec(cols, 2)

    # Y-axis regression:
    model_fit = sfm.OLS(Y, sfm.add_constant(X)).fit()
    model_residuals = model_fit.resid

    for i in range(cols):
        x_name = X.columns[i]
        Y_n = X.iloc[:, i]
        X_n = X.drop(x_name, 1)
        X_model_fit = sfm.OLS(Y_n, sfm.add_constant(X_n)).fit()
        X_model_residuals = X_model_fit.resid
        X_model_R2 = X_model_fit.rsquared  # R2 of Xi v all other Xi
        ax1 = plt.subplot(gs[i, 0])
        sns.regplot(X_model_residuals, model_residuals, ax=ax1, logistic=logistic_, logx=logx_)
        ax1.set_title('')
        ax1.set_xlabel(x_name + ' | All others Xi => R2 = ' + str(round(X_model_R2, 1)))
        ax1.set_ylabel('Y | All others Xi')
        ylim = ax1.get_ylim()
        if i != 0:
            ax1.set_xticklabels([''])
        else:
            ax1.set_title('Residuals Comparison \n', size=15)
            for tick in ax1.xaxis.get_major_ticks():
                tick.label1On = False
                tick.label2On = True
    plt.show()

#####################################################################################################


def vif_info_clean(df, thresh=5, clean_df=False):
    '''
    Calculates VIF each feature in a pandas dataframe and user decided whether or not it wants a
    a clean dataframe with those features with VIF>threshold removed.
    A constant is added to variance_inflation_factor or the results will be incorrect:
    Params
    ----------------
    df =  dataframe
    thresh =  maximum VIF value before the feature is removed from the dataframe
    clean_df = False default. If "True" it returns a dataframe with VIF>threshold vars removed.
    '''

    const = sfm.add_constant(df)  # adding constant
    cols = const.columns
    variables = np.arange(const.shape[1])
    vif_df = pd.Series([sm.stats.outliers_influence.variance_inflation_factor(const.values, i)
                        for i in range(const.shape[1])],
                       index=const.columns).to_frame()

    vif_df = vif_df.sort_values(by=0, ascending=False).rename(columns={0: 'VIF'})
    vif_df_x = vif_df.drop('const')
    vif_df_t = vif_df_x[vif_df['VIF'] > thresh]

    print('Features VIF > threshold: \n')
    print('-*25')
    print(vif_df_t)

    if clean_df == False:
        return vif_df
    else:
        col_to_drop = list(vif_df_t.index)
        print('dropping : ', col_to_drop)
        df = df.drop(col_to_drop, 1)
        return df

 #############################################################################################
 def dummy_generator(dataframe,vars_dum):
    '''
    Creates new dummy variables and returns a new dataframe with original data plus new dummy variables. 
    First variable for each dummy variable is deleted in order to have a ready predictor object for a model
    dataframe = features
    vars_dum = string or list of string with dataframe cols to be dummified.
    '''
    import pandas as pd
    import numpy as np
    df_n = pd.get_dummies(dataframe,columns=vars_dum,prefix=vars_dum, prefix_sep='__')
    drop_var_lst =[i+'__'+str(sorted(dataframe[i])[0]) for i in vars_dum] 
    df_n.drop(drop_var_lst,1, inplace = True)
    return df_n

##################################################################################################################
def LDA_plot_classes(y,x, joint_plot = True):
    '''
    Returns key stats and normality tests as well as probability density functions.
    Useful to understand whether LDA (Linear Discrimination) assumptions of feature normality per class and 
    same variance can be relied upon or we need to transform the data.
    
    Params
    -------
    y = series containing labelled series
    x = feature related to y
    joint_plot = 'True' defaults. To obtain separate plots use 'False'
   
    '''
    labels = list(set(y))
    df = pd.DataFrame()
  
    for lab in labels:
        # Subset to the airline
        subset = x[y == lab]
        # Draw the density plot
        sns.distplot(subset, hist = False, kde = True,
                     kde_kws = {'linewidth': 3},
                     label = lab)
        mean = np.mean(subset)
        std = np.std(subset)
        skew = stats.skew(subset)
        kurt = stats.kurtosis(subset, fisher=False)
        JB_test = list(stats.jarque_bera(subset))
        SW_test = list(stats.shapiro(subset))
        DA_test = list(stats.normaltest(subset))
        df[lab] = pd.Series([mean, std, skew, kurt,
                                          JB_test[1], SW_test[1], DA_test[1], np.mean([JB_test[1], SW_test[1], DA_test[1]])], 
                                         index=['mean', 'std', 'skew', 'kurt',
                                                'JB_test_pv', 'SW_test_pv', 'DA_test_pv','mean_pv'])
        if joint_plot!=True:
            plt.show()
    print('Ho: Normality')
    return df 
