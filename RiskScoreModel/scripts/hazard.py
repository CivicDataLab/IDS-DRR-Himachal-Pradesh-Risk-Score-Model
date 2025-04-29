import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm 
import os
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")
path = os.getcwd() + r"/HP/flood-data-ecosystem-Himachal-Pradesh"

master_variables = pd.read_csv(path+'/RiskScoreModel/data/MASTER_VARIABLES.csv')

hazard_vars = ['inundation_intensity_mean_nonzero', 'inundation_intensity_sum', 'mean_rain', 'max_rain','drainage_density', 'Sum_Runoff', 'Peak_Runoff','slope_mean','elevation_mean','distance_from_river_mean']

hazard_df = master_variables[hazard_vars + ['timeperiod', 'object_id']]

hazard_df_months = []
for month in tqdm(hazard_df.timeperiod.unique()):
    hazard_df_month = hazard_df[hazard_df.timeperiod == month]
    
    # Define the corresponding categories
    #categories = ['very low', 'low', 'medium', 'high', 'very high']
    categories = [1, 2, 3, 4, 5]
    categories_reversed = [5, 4, 3, 2, 1]


    mean = hazard_df_month['distance_from_river_mean'].mean()
    std = hazard_df_month['distance_from_river_mean'].std()
    
    # Define the conditions for each category
    conditions = [
        (hazard_df_month['distance_from_river_mean'] <= mean),
        (hazard_df_month['distance_from_river_mean'] > mean) & (hazard_df_month['distance_from_river_mean'] <= mean + std),
        (hazard_df_month['distance_from_river_mean'] > mean + std) & (hazard_df_month['distance_from_river_mean'] <= mean + 2 * std),
        (hazard_df_month['distance_from_river_mean'] > mean + 2 * std) & (hazard_df_month['distance_from_river_mean'] <= mean + 3 * std),
        (hazard_df_month['distance_from_river_mean'] > mean + 3 * std)
    ]
    # Create the new column based on the conditions
    hazard_df_month['distance_from_river_mean'] = np.select(conditions, categories_reversed, default='outlier')


    # Calculate mean and standard deviation
    mean = hazard_df_month['drainage_density'].mean()
    std = hazard_df_month['drainage_density'].std()
    
    # Define the conditions for each category
    conditions = [
        (hazard_df_month['drainage_density'] <= mean),
        (hazard_df_month['drainage_density'] > mean) & (hazard_df_month['drainage_density'] <= mean + std),
        (hazard_df_month['drainage_density'] > mean + std) & (hazard_df_month['drainage_density'] <= mean + 2 * std),
        (hazard_df_month['drainage_density'] > mean + 2 * std) & (hazard_df_month['drainage_density'] <= mean + 3 * std),
        (hazard_df_month['drainage_density'] > mean + 3 * std)
    ]
    # Create the new column based on the conditions
    hazard_df_month['drainage_density_level'] = np.select(conditions, categories, default='outlier')
    
    #!! ************** !!#
    # Calculate mean and standard deviation
    mean = hazard_df_month['mean_rain'].mean()
    std = hazard_df_month['mean_rain'].std()
    
    # Define the conditions for each category
    conditions = [
        (hazard_df_month['mean_rain'] <= mean),
        (hazard_df_month['mean_rain'] > mean) & (hazard_df_month['mean_rain'] <= mean + std),
        (hazard_df_month['mean_rain'] > mean + std) & (hazard_df_month['mean_rain'] <= mean + 2 * std),
        (hazard_df_month['mean_rain'] > mean + 2 * std) & (hazard_df_month['mean_rain'] <= mean + 3 * std),
        (hazard_df_month['mean_rain'] > mean + 3 * std)
    ]
    # Create the new column based on the conditions
    hazard_df_month['mean_rain_level'] = np.select(conditions, categories, default='outlier')
    #!! ************** !!#
    # Calculate mean and standard deviation
    mean = hazard_df_month['max_rain'].mean()
    std = hazard_df_month['max_rain'].std()
    
    # Define the conditions for each category
    conditions = [
        (hazard_df_month['max_rain'] <= mean),
        (hazard_df_month['max_rain'] > mean) & (hazard_df_month['max_rain'] <= mean + std),
        (hazard_df_month['max_rain'] > mean + std) & (hazard_df_month['max_rain'] <= mean + 2 * std),
        (hazard_df_month['max_rain'] > mean + 2 * std) & (hazard_df_month['max_rain'] <= mean + 3 * std),
        (hazard_df_month['max_rain'] > mean + 3 * std)
    ]
    # Create the new column based on the conditions
    hazard_df_month['max_rain_level'] = np.select(conditions, categories, default='outlier')
    #!! ************** !!#
    
    # Calculate mean and standard deviation
    mean = hazard_df_month['inundation_intensity_mean_nonzero'].mean()
    std = hazard_df_month['inundation_intensity_mean_nonzero'].std()
    
    # Define the conditions for each category
    conditions = [
        (hazard_df_month['inundation_intensity_mean_nonzero'] <= mean),
        (hazard_df_month['inundation_intensity_mean_nonzero'] > mean) & (hazard_df_month['inundation_intensity_mean_nonzero'] <= mean + std),
        (hazard_df_month['inundation_intensity_mean_nonzero'] > mean + std) & (hazard_df_month['inundation_intensity_mean_nonzero'] <= mean + 2 * std),
        (hazard_df_month['inundation_intensity_mean_nonzero'] > mean + 2 * std) & (hazard_df_month['inundation_intensity_mean_nonzero'] <= mean + 3 * std),
        (hazard_df_month['inundation_intensity_mean_nonzero'] > mean + 3 * std)
    ]
    # Create the new column based on the conditions
    hazard_df_month['inundation_intensity_mean_nonzero_level'] = np.select(conditions, categories, default='outlier')
    #!! ************** !!#

    # Calculate mean and standard deviation
    mean = hazard_df_month['inundation_intensity_sum'].mean()
    std = hazard_df_month['inundation_intensity_sum'].std()

    # Define the conditions for each category
    conditions = [
        (hazard_df_month['inundation_intensity_sum'] <= mean),
        (hazard_df_month['inundation_intensity_sum'] > mean) & (hazard_df_month['inundation_intensity_sum'] <= mean + std),
        (hazard_df_month['inundation_intensity_sum'] > mean + std) & (hazard_df_month['inundation_intensity_sum'] <= mean + 2 * std),
        (hazard_df_month['inundation_intensity_sum'] > mean + 2 * std) & (hazard_df_month['inundation_intensity_sum'] <= mean + 3 * std),
        (hazard_df_month['inundation_intensity_sum'] > mean + 3 * std)
    ]
    
    # Create the new column based on the conditions
    hazard_df_month['inundation_intensity_sum_level'] = np.select(conditions, categories, default='outlier')
    #!! ************** !!#

    mean = hazard_df_month['Sum_Runoff'].mean()
    std = hazard_df_month['Sum_Runoff'].std()

    # Define the conditions for each category
    conditions = [
        (hazard_df_month['Sum_Runoff'] <= mean),
        (hazard_df_month['Sum_Runoff'] > mean) & (hazard_df_month['Sum_Runoff'] <= mean + std),
        (hazard_df_month['Sum_Runoff'] > mean + std) & (hazard_df_month['Sum_Runoff'] <= mean + 2 * std),
        (hazard_df_month['Sum_Runoff'] > mean + 2 * std) & (hazard_df_month['Sum_Runoff'] <= mean + 3 * std),
        (hazard_df_month['Sum_Runoff'] > mean + 3 * std)
    ]
    
    # Create the new column based on the conditions
    hazard_df_month['Sum_Runoff'] = np.select(conditions, categories, default='outlier')


    mean = hazard_df_month['slope_mean'].mean()
    std = hazard_df_month['slope_mean'].std()

    # Define the conditions for each category
    conditions = [
        (hazard_df_month['slope_mean'] <= mean),
        (hazard_df_month['slope_mean'] > mean) & (hazard_df_month['slope_mean'] <= mean + std),
        (hazard_df_month['slope_mean'] > mean + std) & (hazard_df_month['slope_mean'] <= mean + 2 * std),
        (hazard_df_month['slope_mean'] > mean + 2 * std) & (hazard_df_month['slope_mean'] <= mean + 3 * std),
        (hazard_df_month['slope_mean'] > mean + 3 * std)
    ]
    
    # Create the new column based on the conditions
    hazard_df_month['slope_mean'] = np.select(conditions, categories, default='outlier')

    
    mean = hazard_df_month['elevation_mean'].mean()
    std = hazard_df_month['elevation_mean'].std()

    # Define the conditions for each category
    conditions = [
        (hazard_df_month['elevation_mean'] <= mean),
        (hazard_df_month['elevation_mean'] > mean) & (hazard_df_month['elevation_mean'] <= mean + std),
        (hazard_df_month['elevation_mean'] > mean + std) & (hazard_df_month['elevation_mean'] <= mean + 2 * std),
        (hazard_df_month['elevation_mean'] > mean + 2 * std) & (hazard_df_month['elevation_mean'] <= mean + 3 * std),
        (hazard_df_month['elevation_mean'] > mean + 3 * std)
    ]
    
    # Create the new column based on the conditions
    hazard_df_month['elevation_mean'] = np.select(conditions, categories_reversed, default='outlier')

    #Average of all levels
    hazard_df_month['flood-hazard'] = (#hazard_df_month['inundation_intensity_mean_nonzero_level'].astype(int)
                                        #+ hazard_df_month['inundation_intensity_sum_level'].astype(int)
                                        hazard_df_month['mean_rain_level'].astype(int)
                                        + hazard_df_month['max_rain_level'].astype(int)
                                        + hazard_df_month['drainage_density_level'].astype(int)
                                        + hazard_df_month['Sum_Runoff'].astype(int)
                                        + hazard_df_month['elevation_mean'].astype(int)
                                        + hazard_df_month['distance_from_river_mean'].astype(int)
                                        + hazard_df_month['slope_mean'].astype(int))/7
    
    hazard_df_month['flood-hazard'] = round(hazard_df_month['flood-hazard'])

    hazard_df_months.append(hazard_df_month)

hazard = pd.concat(hazard_df_months)

master_variables = master_variables.merge(hazard[['timeperiod', 'object_id', 'flood-hazard']],
                       on = ['timeperiod', 'object_id'])

master_variables.to_csv(path+r'/RiskScoreModel/data/factor_scores_l1_flood-hazard.csv', index=False)