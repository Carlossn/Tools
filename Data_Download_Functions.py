###################### Download Data Functions ############################
###### Functions created per Asset Class or source
# These functions are user_friendly versions from http://pandas-datareader.readthedocs.io/en/latest/remote_data.html

# Yahoo Data: fixed using fix_yahoo_finance package:
def Data_Yahoo(ticker,start,end, source='yahoo',price_only=True):
    from pandas_datareader import data
	import pandas as pd
	import fix_yahoo_finance as yf 
	yf.pdr_override() 
    
    if price_only==True:
        df = data.get_data_yahoo(ticker, start, end)
        df= df['Adj Close']
    else:
        df = data.get_data_yahoo(ticker, start, end)

    df.index.rename(None, inplace=True)
    return df

# Quandl Data: using Quandl libary => NOT FINISHED
def Data_Quandl(symbol,start,end, address, key_file_name, ql_source='WIKI/PRICES'):
    '''
    ql_source = select quandl source. US Stocks (WIKI/PRICES) by default
    address = 'c/folder_name/.../folder_name', select folder path for API_key.ini  
    key_file_name = .ini file located in "address" path that contains API key with the next format:
     	['Quandl']
     	api_key = private_key_provided_by_quandl
    '''
    import configparser	
    import quandl
    import os
    os.chdir(address)
    config = configparser.ConfigParser()
    config.read(key_file_name.decode())
    key = config['Quandl']['api_key'].encode()
    quandl.ApiConfig.api_key = key
	df = quandl.get_table('WIKI/PRICES', ticker = [symbol],   # change WIKI/PRICES for other sources
                                date = { 'gte': start, 'lte': end }, 
                                paginate=True)
    return df

# FRED Data: Federal Reserve Data 
def Data_FRED(symbol, start, end, address,key_file_name, revised_data=True):
	'''
    Function works thanks to fredapi library: pip install fredapi 
    revised_data = True by default. To obtain first release figures type False.
    search series id in https://fred.stlouisfed.org/tags or using fred.search('generic_name')
    More fred functions in: https://github.com/mortada/fredapi 
    More fred examples in:  http://mortada.net/python-api-for-fred.html 
    '''
    from fredapi import Fred
    import  pandas as pd
	import datetime
    import configparser 
    import os
    os.chdir(address)
    config = configparser.ConfigParser()
    config.read(key_file_name.decode())
    key = config['FRED']['api_key'].encode()
    fred = Fred(api_key=key)
    if revised_data == True:
        df = fred.get_series(symbol, start, end)
	else:
        df = fred.get_series_first_release(symbol)
        df = df[(df.index >= start) & (df.index <= end)]
    return df

# Fama-French Data: Federal Reserve Data 
def Data_FF(series_name,start,encode):
    '''
    data info in http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
    type get_available_datasets() to get series_name possibilities
    '''
    from pandas_datareader.famafrench import get_available_datasets
    import pandas_datareader.data as web
    import pandas as pd
    df = web.DataReader(series_name, 'famafrench',start,end)[0]
    df.index.rename(None,inplace=True)
    return df

def Data_FF_search(term,all_info=False):
    '''
    Search specific Summary of sets available to download using DataReader_FF function
    '''
    from pandas_datareader.famafrench import get_available_datasets
    import pandas_datareader.data as web
    import numpy as np
    import pandas as pd
    sets = np.array(get_available_datasets())
    if all_info!=False:
        data = sets
    else:
        b_idx = np.array([x.find(term)!=-1 for x in sets])
        data = sets[b_idx]
    return data

# World Bank Data: source World Bank
def Data_WB(series_name, country_list, start,end):
    '''
    Download series from WB database
    For searching series names using generic names do: wb.search('gdp.*capita.')
    For checking country codes do: wb.country_codes
    More info about wb functions in http://pandas-datareader.readthedocs.io/en/latest/remote_data.html#world-bank
    More info about WB data series: https://data.worldbank.org/
    '''
    from pandas_datareader import wb
    import numpy as np
    import pandas as pd
    data = wb.download(indicator=series_name, country=country_list, start=start, end=end)
    return data

# World Bank Data: source World Bank
def Data_Eurostat(series_name, start,end):
    '''
    Download series from Eurostat database
    For searching series names using generic names visit http://ec.europa.eu/eurostat/data/browse-statistics-by-theme
    More info about Eurostat functions in http://pandas-datareader.readthedocs.io/en/latest/remote_data.html#eurostat
    Output is Multiindex object. To dig into country details only index using data[level1][level2]...['Country_Name] 
    '''
    from pandas_datareader import wb
    import numpy as np
    import pandas as pd
    data = web.DataReader(series_name, 'eurostat',start,end)
    return data
 