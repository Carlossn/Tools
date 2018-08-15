import numpy as np
import pandas as pd
import statsmodels as sm
%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.graphics.gofplots import ProbPlot
import warnings


################################# OLS SIMPLE MODEL TOOLS#######################################################################

def OLS_Assumption_Tests(x, y):
    '''
    Helpful to understand whether or not the model meets the traditional 
    OLS model assumptions: linearity, normality, homoskedasticity and non-autocorrelation
    x = explanatory vars/features array/df
    y = dependent var array

    '''

    # fitting model:
    model_fit = sm.api.formula.OLS(y, sm.api.add_constant(x)).fit()  # new regression fit as it works with sm.fit() only
    # model residuals
    model_residuals = model_fit.resid
    # TESTS:
    # linearity Tests:
    HG_test = sm.stats.diagnostic.linear_harvey_collier(model_fit)  # Ho Linearity, Harvey-Collier Test
    # Normality Tests:
    JB_test = stats.jarque_bera(model_residuals)  # Ho Normality, Jarque-Bera Test
    # Homocedasticity Tests (Ho: Homocedasticity)
    BP_test = sm.stats.diagnostic.het_breuschpagan(model_residuals, x.reshape(-1, 1))[2:4]  # Breusch-Pagan Lagrange Multiplier
    W_test = sm.stats.diagnostic.het_white(model_residuals, sm.api.add_constant(x))[2:4]  # White test
    GQ_test = sm.stats.diagnostic.het_goldfeldquandt(model_residuals, sm.api.add_constant(x))[0:2]  # Ho diff here: Hetereoced= Var(resides)=Var(X)
    # Non-Autocorrel Tests (Ho: No autocorrelation resids)
    DW_stat = sm.stats.stattools.durbin_watson(model_residuals)
    DW_test = [DW_stat, np.NAN]  # Ho: No Autocorrelation  ,Durbin-Watson
    LJB_output = sm.stats.diagnostic.acorr_ljungbox(model_residuals, lags=int(round(np.log(len(model_residuals)), 0)))  # if Lags=None => default maxlag= ‘min((nobs // 2 - 2), 40)’ # Lags=None => default maxlag= ‘min((nobs // 2 - 2), 40)
    LJB_test = [np.max(LJB_output[0]), np.min(LJB_output[1])]
    BG_test = sm.stats.diagnostic.acorr_breusch_godfrey(model_fit, nlags=int(round(np.log(len(model_residuals)), 0)))[2:4]  # Breusch Godfrey Lagrange Multiplier tests
    # Summary DF:
    df = pd.DataFrame([HG_test, JB_test, BP_test, W_test, GQ_test, DW_test, LJB_test, BG_test],
                      columns=['statistic', 'pvalue'],
                      index=['HG Test - Ho: Linearity', 'JB Test - Ho: Normality',
                             'BP Test - Ho: Homoced', 'W Test - Ho: Homoced', 'GQ Test - Ho: Heteroced',
                             'DW Test - Ho: Non-Autocorrel', 'LJB Test - Ho: Non-Autocorrel', 'BG Test - Ho: Non-Autocorrel'])
    return df

############################################################################################################################


def OLS_Assumptions_Plot(x, y, resid_type='norm'):
    '''
    Plotting key charts to check OLS assumptions: linearity, normality, homoskedasticity and non-autocorrelation
    '''
    # Calculations:
    # fitting model:
    model_fit = sm.formula.api.OLS(y, sm.api.add_constant(x)).fit()  # new regression fit as it works with sm.fit() only
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

    def heteroced_plot(x, y, resid_type='abs_sq_norm'):
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
    linearity_plot(x, y)
    ax2 = plt.subplot(222)
    normality_plot(model_residuals)
    ax3 = plt.subplot(223)
    heteroced_plot(x, y, resid_type)
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

    %matplotlib inline
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import statsmodels as sm
    from scipy import stats
    from statsmodels.graphics.gofplots import ProbPlot
    import warnings

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
