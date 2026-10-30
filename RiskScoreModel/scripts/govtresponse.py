import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm 
import os
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")
path = os.getcwd() #+ r"/HP/flood-data-ecosystem-Himachal-Pradesh"
master_variables = pd.read_csv(path+'/RiskScoreModel/data/MASTER_VARIABLES.csv')

def get_financial_year(timeperiod):
    if int(timeperiod.split('_')[1]) >= 4:
        return str(int(timeperiod.split('_')[0]))+'-'+str(int(timeperiod.split('_')[0])+1)
    else:
        return str(int(timeperiod.split('_')[0]) - 1)+'-'+str(int(timeperiod.split('_')[0]))
    
# Apply the function to create the 'FinancialYear' column
master_variables['FinancialYear'] = master_variables['timeperiod'].apply(lambda x: get_financial_year(x))

#INPUT VARS
government_response_vars = ["total_tender_awarded_value",
                       # 'Repair and Restoration_tenders_awarded_value',
                       # 'LWSS_tenders_awarded_value', 'NDRF_tenders_awarded_value', 
                       # 'SDMF_tenders_awarded_value', 'WSS_tenders_awarded_value', 
                       # 'Preparedness Measures_tenders_awarded_value', 
                       # 'Immediate Measures_tenders_awarded_value', 
                       # "Others_tenders_awarded_value",
                       'relief_and_mitigation_sanction_value'
                      ]

# Find cumsum in each FY of the government response vars
# (on a copy — master_variables must keep raw monthly values, since the saved
# factor file is merged with the other factor files downstream)
govtresponse_df = master_variables[government_response_vars + ['timeperiod', 'object_id']].copy()
for var in government_response_vars:
    govtresponse_df[var] = master_variables.groupby(['object_id','FinancialYear'])[var].cumsum()


# How the cumulative spend becomes the 1-5 class. 5 is the WORST response and
# therefore the highest contribution to risk; 1 is the best.
#
#   'zband' - symmetric z-score bands on log spend, pooled over all months
#   'sigma' - the original one-sided ladder (kept so the old numbers can be
#             reproduced), which does not work on this data:
#
# The old rule put every tehsil at or below the monthly MEAN into class 5, and
# reserved classes 2 and 1 for tehsils more than 2 and 3 standard deviations
# ABOVE it. Tender spend is heavily right-skewed - per-month skew of the scaled
# sum has a median of 2.68 and the mean sits 2.16x the median - so "at or below
# the mean" is the majority of tehsils: 64.8% of HP tehsil-months landed in
# class 5 (90.5% in Bihar), while class 1 was mathematically almost unreachable
# at 1.4% and class 2 at 2.0%. Twelve HP tehsils were class 5 in all 66 months.
# The whole range from "received nothing at all" to "received just under the
# average" collapsed into one class, which is where the discrimination went.
#
# Taking log1p first makes the spread roughly symmetric, so the band edges land
# inside the data instead of out in the tail, and centring the bands on the mean
# instead of starting at it uses both halves of the distribution. Pooling the
# fit over all months (rather than refitting per month) keeps the class absolute:
# a month in which no tehsil was funded shows up as every tehsil scoring 5,
# instead of being rescaled into a spread that implies some tehsils did well.
RESPONSE_SCALE = 'zband'   # 'zband' | 'sigma'

# Band edges in standard deviations from the mean, worst-to-best.
ZBAND_EDGES = (-1.5, -0.5, 0.5, 1.5)


def _sigma_ladder(score):
    """Original one-sided ladder: <= mean is 5, then +1 sd steps down to 1."""
    mean = score.mean()
    std = score.std()
    conditions = [
        (score <= mean),
        (score > mean) & (score <= mean + std),
        (score > mean + std) & (score <= mean + 2 * std),
        (score > mean + 2 * std) & (score <= mean + 3 * std),
        (score > mean + 3 * std)
    ]
    return np.select(conditions, [5, 4, 3, 2, 1])


def _z_bands(score):
    """Symmetric bands: below -1.5 sd is 5 (worst), above +1.5 sd is 1 (best)."""
    mean = score.mean()
    std = score.std()
    if not np.isfinite(std) or std == 0:
        return np.full(len(score), 3)
    lo, mlo, mhi, hi = [mean + e * std for e in ZBAND_EDGES]
    conditions = [score <= lo, score <= mlo, score <= mhi, score <= hi, score > hi]
    return np.select(conditions, [5, 4, 3, 2, 1])


if RESPONSE_SCALE == 'zband':
    # log1p pulls in the right tail; zeros stay pinned at the bottom, so class 5
    # ends up meaning "this tehsil received nothing this financial year to date".
    logged = np.log1p(govtresponse_df[government_response_vars])
    # z-score each variable over the pooled set so the two are commensurate,
    # then average rather than sum - a sum would let the wider-spread variable
    # dominate purely because of its spread.
    z = (logged - logged.mean()) / logged.std()
    govtresponse_df['sum'] = z.mean(axis=1)
    govtresponse_df['government-response'] = _z_bands(govtresponse_df['sum'])
    govtresponse_df_months = [govtresponse_df]
else:
    govtresponse_df_months = []
    for month in tqdm(govtresponse_df.timeperiod.unique()):
        govtresponse_df_month = govtresponse_df[govtresponse_df.timeperiod == month]
        # Initialize MinMaxScaler
        scaler = MinMaxScaler()
        # Fit scaler to the data and transform it
        govtresponse_df_month[government_response_vars] = scaler.fit_transform(govtresponse_df_month[government_response_vars])

        # Sum scaled exposure vars
        govtresponse_df_month['sum'] = govtresponse_df_month[government_response_vars].sum(axis=1)

        govtresponse_df_month['government-response'] = _sigma_ladder(govtresponse_df_month['sum'])

        govtresponse_df_months.append(govtresponse_df_month)

govtresponse = pd.concat(govtresponse_df_months)
master_variables = master_variables.merge(govtresponse[['timeperiod', 'object_id', 'government-response']],
                       on = ['timeperiod', 'object_id'])

master_variables.to_csv(path+'/RiskScoreModel/data/factor_scores_l1_government-response.csv', index=False)