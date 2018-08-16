import numpy as np
import pandas as pd
import statsmodels as sm
import statsmodels.api as sfm
%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.graphics.gofplots import ProbPlot
import warnings
import matplotlib.gridspec as gridspec
import matplotlib as mpl


################################# LR MODEL TOOLS##########################################################
# Functions:
# OLS_Assumption_Tests: check OLS model assumptions
# OLS_Assumptions_Plot: plot OLS model assumptions
# influence_cook_plot: spot points with outlier residuals (outliers) and high leverage that distort the model
# cook_dist_plot: Cook distance measures the effect of deleting a given observation over the regression fit.
# corr_mtx_des: description and correlation matrix to spot features that display multicollinearity
# multivar_LR_plot: plots multiple univariate regressions vs Y to confirm if a univariate OLS couldd be
# sufficient to explain the response variable.

##################################################################################################################

def LR_Assumptions_Tests(x, y):
    '''
    Helpful to understand whether or not the model meets the traditional 
    OLS model assumptions: linearity, normality, homoskedasticity and non-autocorrelation
    x = explanatory vars/features array/df
    y = dependent var array
    '''
    import numpy as np
    import pandas as pd
    import statsmodels as sm
    import statsmodels.api as sfm

    # unvariate or multivar LR:
    try:
        x.columns
    except:
        univariate = True
    else:
        univariate = False

    # fitting model:
    model_fit = sfm.OLS(y, sfm.add_constant(x)).fit()  # new regression fit as it works with sm.fit() only
    # model residuals
    model_residuals = model_fit.resid
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

    %matplotlib inline
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import statsmodels as sm
    import statsmodels.api as sfm
    from scipy import stats
    from statsmodels.graphics.gofplots import ProbPlot
    import warnings

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
    import matplotlib.gridspec as gridspec
    import matplotlib as mpl
    import matplotlib.pyplot as plt

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
