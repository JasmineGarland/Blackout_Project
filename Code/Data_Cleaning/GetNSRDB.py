#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#from nsrdb import *
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from io import StringIO
import requests

from concurrent.futures import as_completed
from copy import deepcopy
import datetime
import h5py
from itertools import groupby
import json
import logging
import numpy as np
import os
import pandas as pd
import shutil
import os
import pandas as pd
import glob



# In[ ]:


def is_leap_year(year):
    leap_year = False
    year = float(year)
    if (year % 4) == 0:
        if (year % 100) == 0:
            if (year % 400) == 0:
                leap_year = True
            else:
                leap_year = False
        else:
            leap_year = True
    else:
        leap_year = False
    return leap_year






#get solar data from NSRDB
def get_solar_data(lat, lon, year, api_key, info):
    #attributes = 'ghi,dhi,dni,clearsky_dni,total_precipitable_water,surface_albedo'
    attributes = 'clearsky_dni,surface_pressure,total_precipitable_water,wind_direction,air_temperature,relative_humidity,dew_point,ghi,dhi,dni,wind_speed,air_temperature,surface_albedo'

    if is_leap_year(year):
        leap_year = 'true'
    else:
        leap_year = 'false'
    #
    interval = '60'
    utc = 'false'
    your_name = info[0]
    reason_for_use = info[1]
    your_affiliation = info[2]
    your_email = info[3]
    mailing_list = 'false'
    url = 'https://developer.nrel.gov/api/solar/nsrdb_psm3_download.csv?wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap}&interval={interval}&utc={utc}&full_name={name}&email={email}&affiliation={affiliation}&mailing_list={mailing_list}&reason={reason}&api_key={api}&attributes={attr}'.format(year=year, lat=lat, lon=lon, leap=leap_year, interval=interval, utc=utc, name=your_name, email=your_email, mailing_list=mailing_list, affiliation=your_affiliation, reason=reason_for_use, api=api_key, attr=attributes)
    # Return just the first 2 lines to get metadata:
    solar_data = pd.read_csv(StringIO(requests.get(url).text), skiprows=2)
    solar_data['sky_cover'] = (1 - solar_data['DNI']/solar_data['Clearsky DNI'])*10
    solar_data['sky_cover'][solar_data['Clearsky DNI']==0] = 5
    solar_data['date_time'] = solar_data['Year'].astype(str) + '-' + solar_data['Month'].astype(str) + '-'+ solar_data['Day'].astype(str)+ ' ' + solar_data['Hour'].astype(str) + ':'+ solar_data['Minute'].astype(str)
    solar_data['date_time'] = pd.to_datetime(solar_data['date_time'])
    #convert pressure from Millibar to PA
    solar_data['Pressure'] =  solar_data['Pressure']*100
    solar_data.index = solar_data['date_time']
    solar_data.drop(['date_time'], axis=1, inplace=True)
    #solar_data.drop(['Year'], axis=1, inplace=True)
    #solar_data['Year'] = 2010
    #solar_data.drop(['Day'], axis=1, inplace=True)
    #solar_data.drop(['Month'], axis=1, inplace=True)
    #solar_data.drop(['Hour'], axis=1, inplace=True)
    #solar_data.drop(['Minute'], axis=1, inplace=True)
    #solar_data_p = solar_data.resample('H').mean()
    #data_info = pd.read_csv('/Users/jgarland/Desktop/comstock-weather-main/resources/counties_centroids_twolocations.csv')
    # print(solar_data_p)
    return solar_data
    return data_info


# In[ ]:


#where zone is the iso, reeds ba, or emission areas, should be a string
def get_data_nsrdb(data_info, zone, Path, api_key, info):
    years = [2017, 2018, 2019, 2020, 2021]

    #locations_lat = list(data_info['INTPTLAT'])
    #locations_lon = list(data_info['INTPTLON']) 

    locations_lat = list(data_info['LATITUDE'])
    locations_lon = list(data_info['LONGITUDE'])


    data_years = []
    data = pd.DataFrame()
    df_years = pd.DataFrame()

    #uses the get_solar_data function to get the nsrdb from the API
    for i in range(len(years)):
        for j in range(len(locations_lat)):
            data = get_solar_data(locations_lat[j],locations_lon[j], years[i], api_key, info)
            #data['Location'] = data_info['NAMELSAD'][j]
            data['Location'] = data_info['NAME'][j]
            #print(data)
            data_years.append(data)
            df = pd.concat(data_years, axis=0, ignore_index=False) 

        group = df.groupby(['Location'])

        #may want a dict with all df's in the future?
        #for location,name in group:
            #d[location] = pd.DataFrame(df)

        #for now just create df based off location    
        for location, name in group:
             return exec('{} = pd.DataFrame(name)'.format(location.replace(' ','_')))
          #exec('{} = pd.DataFrame(name)'.format((location))  
             return pd.DataFrame(name).to_csv(Path + "/" + zone +'_'+location+'_NSRDB.csv')




def get_reedsba(path):
    all_files = glob.glob(path + "/*.csv")
    
    li = []
    
    for filename in all_files:
        df = pd.read_csv(filename, header=0)
        li.append(df)
    
    
        #Create one dataframe
    df = pd.concat(li, axis=0, ignore_index=True)
    df['date_time'] = pd.to_datetime(df['date_time'])
    df.set_index('date_time', inplace=True)
    return df



# In[ ]:


def get_iso(path):
    folders = []

    directory = os.path.join(path)
    for root,dirs,files in os.walk(directory):
        folders.append(root)

    del folders[0]


    final = []
    for folder in folders:
        df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(folder + "/*.csv"))))
        final.append(df)

    df = pd.concat(final, axis=0, ignore_index=True)
    df['date_time'] = pd.to_datetime(df['date_time'])
    df.set_index('date_time', inplace=True)
    return df


# In[ ]:


def variables_cdf(df):
    
    '''
    df: a series of the feature you want. 
    '''
    np_loc =  df.to_numpy().reshape(-1, 1)
    cdf_obj = tmy.Cdf(np_loc,df.index)
    #best_year = cdf_obj._best_fs_year()
    fs_scores_all = cdf_obj._fs_stat()
    
    #if you want to plot the CDF's uncomment the 2 lines below
    #for i in range(1,13):
        #cdf_obj.plot_tmy_selection(month=i)
        
    return fs_scores_all


# In[ ]:


def _resample_arr_daily(arr, time_index, fun):
        """Resample an array to daily using a specified function.
        Parameters
        ----------
        arr : np.ndarray
            Array of timeseries data corresponding to time_index.
        time_index : pd.Datetimeindex
            Datetimeindex corresponding to arr.
        fun : str | None
            Resampling method.
        Returns
        -------
        arr : np.ndarray
            Array of daily timeseries data if fun is not None.
        """
        if fun is not None:
            df = pd.DataFrame(deepcopy(arr), index=time_index)
            if 'min' in fun.lower():
                df = df.resample('1D').min()
            elif 'max' in fun.lower():
                df = df.resample('1D').max()
            elif 'mean' in fun.lower():
                df = df.resample('1D').mean()
            elif 'sum' in fun.lower():
                df = df.resample('1D').sum()
            arr = df.values
        return arr


# In[ ]:


def create_df(df_county):    
    data_years = []
    df_min = []
    df_max = []
    df_mean = []
    #df_min_1 = pd.DataFrame(index = pd.date_range(start = '01/01/1998', end='12/31/2020', freq = '1D'))

    #data = pd.DataFrame()
    #df_years = pd.DataFrame(index = df_county.Year.unique())

    for i in df_county.columns[2:13]:
        
        data_daily_min = pd.DataFrame(_resample_arr_daily(df_county[i], df_county.index, 'min'))
        data_daily_max = pd.DataFrame(_resample_arr_daily(df_county[i], df_county.index, 'max'))
        data_daily_mean = pd.DataFrame(_resample_arr_daily(df_county[i], df_county.index, 'mean'))

        
        df_min.append(data_daily_min)
        df_min_1 = pd.concat(df_min, axis=1, ignore_index=False)
        #the cdf call from nsrdb.Tmy needs the index to be the time - reasoning for this, maybe resample from input df instead
        df_min_1.set_index(pd.date_range(start = '01/01/1998', end='12/31/2020', freq = '1D'), inplace=True)
        
        df_max.append(data_daily_max)
        df_max_1 = pd.concat(df_max, axis=1, ignore_index=False)
        df_max_1.set_index(pd.date_range(start = '01/01/1998', end='12/31/2020', freq = '1D'), inplace=True)
        
        df_mean.append(data_daily_mean)
        df_mean_1 = pd.concat(df_mean, axis=1, ignore_index=False)
        df_mean_1.set_index(pd.date_range(start = '01/01/1998', end='12/31/2020', freq = '1D'), inplace=True)
        
       
    names_min = ['min_daily_Clearsky DNI', 'min_daily_Pressure', 'min_daily_Precipitable Water', 'min_daily_Wind Direction',
       'min_daily_Temperature', 'min_daily_Relative Humidity', 'min_daily_Dew Point', 'min_daily_GHI', 'min_daily_DHI', 'min_daily_DNI', 'min_daily_Wind Speed']

    data_min = df_min_1.set_axis(names_min, axis=1, inplace=False)
    
    
    names_max = ['max_daily_Clearsky DNI', 'max_daily_Pressure', 'max_daily_Precipitable Water', 'max_daily_Wind Direction',
       'max_daily_Temperature', 'max_daily_Relative Humidity', 'max_daily_Dew Point', 'max_daily_GHI', 'max_daily_DHI', 'max_daily_DNI', 'max_daily_Wind Speed']

    data_max = df_max_1.set_axis(names_max, axis=1, inplace=False)
    
    names_mean = ['mean_daily_Clearsky DNI', 'mean_daily_Pressure', 'mean_daily_Precipitable Water', 'mean_daily_Wind Direction',
       'mean_daily_Temperature', 'mean_daily_Relative Humidity', 'mean_daily_Dew Point', 'mean_daily_GHI', 'mean_daily_DHI', 'mean_daily_DNI', 'mean_daily_Wind Speed']

    data_mean = df_mean_1.set_axis(names_mean, axis=1, inplace=False)
        
    names = ['Month', 'Year','Clearsky DNI', 'Pressure', 'Precipitable Water', 'Wind Direction',
       'Temperature', 'Relative Humidity', 'Dew Point', 'GHI', 'DHI', 'DNI', 'Wind Speed']
        
    
    #return df_final, data_min, data_max, data_mean
    return data_min, data_max, data_mean


# In[ ]:


def min_max_mean_fs(df,df1, data_info, spat_lookup, index_):
    min_df = df[0]
    max_df = df[1]
    mean_df = df[2]

    list_min = []
    data_obj_min = pd.DataFrame()
    #only the columns we want
    for i in min_df.iloc[:,[4,6,7,9,10]]:
        #this creates dict of arrays for each variable and year
        data_obj_min = variables_cdf(min_df[i])
        #turns each dict of arrays into a dataframe
        df_transform_min = pd.concat([pd.DataFrame(v) for k, v in data_obj_min.items()], axis = 0, keys = list(data_obj_min.keys()))
        list_min.append(df_transform_min)
        #concats the columns together
        df_min = pd.concat(list_min, axis=1, ignore_index=False, keys = list(data_obj_min.keys()))
        df_min.columns = df_min.columns.droplevel()
        df_min.reset_index(inplace = True)  
        
    data_years = []
    data = pd.DataFrame()

    for i in max_df.iloc[:,[4,6,7,9,10]]:
        data = variables_cdf(max_df[i])
        df_year = pd.concat([pd.DataFrame(v) for k, v in data.items()], axis = 0, keys = list(data.keys()))
        data_years.append(df_year)
        df_max = pd.concat(data_years, axis=1, ignore_index=False, keys = list(data.keys()))
        df_max.columns = df_max.columns.droplevel()
        df_max.reset_index(inplace = True)  

    data_years = []
    data = pd.DataFrame()

    for i in mean_df.iloc[:,[4,6,7,9,10]]:
        data = variables_cdf(mean_df[i])
        df_year = pd.concat([pd.DataFrame(v) for k, v in data.items()], axis = 0, keys = list(data.keys()))
        data_years.append(df_year)
        df_mean = pd.concat(data_years, axis=1, ignore_index=False, keys = list(data.keys()))
        df_mean.columns = df_mean.columns.droplevel()
        df_mean.reset_index(inplace = True)  
        
    df_mmm = pd.concat([df_min, df_max, df_mean],axis = 1)
    
    names = ['month', 'year','fs_min_daily_Temperature', 'fs_min_daily_Dew Point', 'fs_min_daily_GHI', 'fs_min_daily_DNI',
    'fs_min_daily_Wind Speed','month', 'year', 'fs_max_daily_Temperature', 'fs_max_daily_Dew Point', 'fs_max_daily_GHI', 'fs_max_daily_DNI',
    'fs_max_daily_Wind Speed','month', 'year','fs_mean_daily_Temperature', 'fs_mean_daily_Dew Point', 'fs_mean_daily_GHI','fs_mean_daily_DNI', 
    'fs_mean_daily_Wind Speed']
    data = pd.DataFrame()

    data = pd.DataFrame(df_mmm.set_axis(names, axis=1, inplace=False))
    data = data.loc[:,~data.columns.duplicated()].copy()
    #years
    year = pd.Series(df1['Year'].unique())
    year_index = pd.concat([year] * 12, axis=0, ignore_index=True)
    data['year'] = (year_index)
   
    data['index'] = int(index_)
    #return data
    data_info['index'] = data_info.index.astype(int)
    
    df_loc = pd.merge(data, data_info.iloc[:, [0,3,4,15,16,17]], on ='index')
    #df_loc = pd.merge(data, data_info.iloc[:, [3,4,16,17]], on ='index')

    df_loc['GEOJOIN'] =df_loc['GEOID'].apply(lambda x: '{0:0>5}'.format(x))
    spat_lookup['GEOJOIN'] = spat_lookup['nhgis_2010_county_gisjoin'].str.slice(1,3) + spat_lookup['nhgis_2010_county_gisjoin'].str.slice(4,7)
    df_final = pd.merge(df_loc,spat_lookup, on = 'GEOJOIN')

    drop = ['weather_file_2012', 'custom_region', 'iecc_2012_climate_zone_2a_split',
       'weather_file_2015', 'weather_file_2016', 'weather_file_2017',
       'weather_file_2018', 'weather_file_2019', 'weather_file_TMY3',
       'source_count', 'housing_units_2020_redistricting',
       'occupied_units_2020_redistricting', 'vacant_units_2020_redistricting',
        'index', 'GEOID', 'vacant_units', 'occupied_units', 'building_america_climate_zone',
        'nhgis_2010_county_gisjoin', 'census_region', 'census_division', 'ahs_region_2013',
        'census_division_recs', 'state_name', 'state_abbreviation']
    
    df_final.drop(columns = drop, inplace = True)


    return df_final


# In[ ]:


def weights(df_final):
    df_final['w_max_Temperature'] = 1/20
    df_final['w_min_Temperature'] = 1/20
    df_final['w_mean_Temperature'] = 2/20

    df_final['w_max_Dewpoint'] = 1/20
    df_final['w_min_Dewpoint'] = 1/20
    df_final['w_mean_Dewpoint'] = 2/20

    df_final['w_max_WindVelocity'] = 1/20
    df_final['w_mean_WindVelocity'] = 1/20

    df_final['w_GHI'] = 5/20
    df_final['w_DNI'] = 5/20

    df_final['weighted_max_Temperature'] = df_final['w_max_Temperature'] * df_final['fs_max_daily_Temperature']
    df_final['weighted_min_Temperature'] = df_final['w_min_Temperature'] * df_final['fs_min_daily_Temperature']
    df_final['weighted_mean_Temperature'] = df_final['w_mean_Temperature'] * df_final['fs_mean_daily_Temperature']

    df_final['weighted_max_Dewpoint'] = df_final['w_max_Dewpoint'] * df_final['fs_max_daily_Dew Point']
    df_final['weighted_min_Dewpoint'] = df_final['w_min_Dewpoint'] * df_final['fs_min_daily_Dew Point']
    df_final['weighted_mean_Dewpoint'] = df_final['w_mean_Dewpoint'] * df_final['fs_mean_daily_Dew Point']


    df_final['weighted_max_WindVelocity'] = df_final['w_max_WindVelocity'] * df_final['fs_mean_daily_Wind Speed']
    df_final['weighted_mean_WindVelocity'] = df_final['w_mean_WindVelocity'] * df_final['fs_mean_daily_Wind Speed']

    df_final['weighted_GHI'] = df_final['w_GHI'] * df_final['fs_mean_daily_GHI']
    df_final['weighted_DNI'] = df_final['w_DNI'] * df_final['fs_mean_daily_DNI']

    df_final['weather_fs'] = df_final.iloc[:, 36:46].sum(axis=1)
    df_final['location_weight']  = df_final['housing_units'].div(df_final['housing_units'].unique().sum(), axis=0)
    df_final['location_fs'] = df_final['weather_fs'] * df_final['location_weight'] 
    return df_final


# In[ ]:


def make_tmy_csv(df, Tmy_selection_df, d, alldfs, Path):
    data_years = []
    data = pd.DataFrame()
    df_years = pd.DataFrame()

    for month, year in Tmy_selection_df.index:
        month_year = d[df].loc[(d[df]['Month'] == month) & (d[df]['Year'] == year)]
        data_years.append(month_year)
        tmy_df = pd.concat(data_years, axis=0, ignore_index=False)
        Location = tmy_df.Location
        pd.DataFrame(tmy_df).to_csv(Path + '/'+'ERCOT'+'_'+tmy_df['Location'][0]+'_tmy.csv')

def TMY_Selection(df_final):
    region_df = df_final.groupby(['month', 'year']).sum().reset_index()
    indx = region_df.groupby('month')['location_fs'].idxmin()
    region_df = region_df.loc[indx]
    Tmy_selection_df = region_df.set_index(['month', 'year'])['location_fs']
    return Tmy_selection_df