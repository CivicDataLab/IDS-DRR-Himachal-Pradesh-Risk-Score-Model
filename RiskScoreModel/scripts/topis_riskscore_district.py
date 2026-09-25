from topsis import Topsis
import pandas as pd
import numpy as np 
import os
import glob

fldhzd_w = 4
exp_w = 1
vul_w = 2
resp_w = 2

## MASTER DATA WITH FACTOR SCORES
print(os.getcwd())
## INPUT: FACTOR SCORES CSV
# sorted: glob order is filesystem-dependent, and the first file is the base
# dataframe whose non-factor columns pass through to the final output
factor_scores_dfs = sorted(glob.glob(os.getcwd()+r'/RiskScoreModel/data/factor_scores_l1*.csv'))

# Select only the columns that exist in both the DataFrame and the list
factors = ['exposure', 'flood-hazard', 'vulnerability', 'government-response']
additional_columns = ['efficiency','flood-hazard-float','landd_score']

merged_df = pd.read_csv(factor_scores_dfs[0])
# Merge successive DataFrames in the list
for df in factor_scores_dfs[1:]:
    df = pd.read_csv(df)
    selected_columns = [col for col in factors if col in df.columns]
    # Create a new DataFrame containing only the selected columns
    df = df[selected_columns + ['object_id', 'timeperiod']]
    merged_df = pd.merge(merged_df, df, on=['object_id', 'timeperiod'], how='inner')
##df = pd.read_csv(os.getcwd()+'/RiskScoreModel/data/factor_scores.csv')

def get_financial_year(timeperiod):
    if int(timeperiod.split('_')[1]) >= 4:
        return str(int(timeperiod.split('_')[0]))+'-'+str(int(timeperiod.split('_')[0])+1)
    else:
        return str(int(timeperiod.split('_')[0]) - 1)+'-'+str(int(timeperiod.split('_')[0]))
    
# Apply the function to create the 'FinancialYear' column
merged_df['financial_year'] = merged_df['timeperiod'].apply(lambda x: get_financial_year(x))


# Ensure sorting for proper cumulative sum
merged_df.sort_values(by=['object_id', 'financial_year', 'timeperiod'], inplace=True)

cumulative_vars = [
    'total_tender_awarded_value',
    'restoration_measures_tenders_awarded_value',
    'lwss_tenders_awarded_value',
    'ndrf_tenders_awarded_value',
    'sdmf_tenders_awarded_value',
    'wss_tenders_awarded_value',
    'preparedness_measures_tenders_awarded_value',
    'immediate_measures_tenders_awarded_value',
    'others_tenders_awarded_value',
    'relief_and_mitigation_sanction_value'
]

for var in cumulative_vars:
    cum_var_name = var + "_fy_cumsum"
    merged_df[cum_var_name] = merged_df.groupby(['object_id', 'financial_year'])[var].cumsum()

df_months = []

for month in merged_df.timeperiod.unique():
    #print(month)

    df_month = merged_df[merged_df.timeperiod==month]

    evaluation_matrix = np.array(df_month[[ 'flood-hazard', 'exposure', 'vulnerability', 'government-response']].values)
    weights = [fldhzd_w,exp_w,vul_w,resp_w]

    criterias = [True, True, True, True]
    # All variables - more is more risk; 'government-response' is in reverse

    t = Topsis(evaluation_matrix, weights, criterias)
    t.calc()
    df_month['TOPSIS_Score'] = t.worst_similarity
    df_month = df_month.sort_values(by='TOPSIS_Score', ascending=False)
    
    compositescorelabels = [1,2,3,4,5]
    compscore = pd.cut(df_month['TOPSIS_Score'],bins = 5,precision = 0,labels = compositescorelabels )
    df_month['risk-score'] = compscore

    df_months.append(df_month)

topsis = pd.concat(df_months)

topsis = topsis.drop('District',axis=1)

topsis.columns = [col.lower().replace('_', '-').replace(' ', '-') for col in topsis.columns]
print(topsis.columns)
topsis.to_csv(os.getcwd()+r'/RiskScoreModel/data/risk_score.csv', index=False)

# ## DISTRICT LEVEL SCORES
dist_ids = pd.read_csv(os.getcwd()+r'/RiskScoreModel/assets/district_objectid.csv')

compositescorelabels = ['1','2','3','4','5']

dist_vul = topsis.groupby(['district','timeperiod'])['vulnerability'].sum().reset_index()
compscore = pd.cut(dist_vul['vulnerability'],bins = 5,precision = 0,labels = compositescorelabels )
dist_vul['vulnerability'] = compscore
dist_vul = dist_vul.merge(dist_ids, on='district')

dist_exp = topsis.groupby(['district','timeperiod'])['exposure'].sum().reset_index()
compscore = pd.cut(dist_exp['exposure'],bins = 5,precision = 0,labels = compositescorelabels )
dist_exp['exposure'] = compscore
dist_exp = dist_exp.merge(dist_ids, on='district')

dist_govt = topsis.groupby(['district','timeperiod'])['government-response'].sum().reset_index()
compscore = pd.cut(dist_govt['government-response'],bins = 5,precision = 0,labels = compositescorelabels )
dist_govt['government-response'] = compscore
dist_govt = dist_govt.merge(dist_ids, on='district')

dist_haz = topsis.groupby(['district','timeperiod'])['flood-hazard'].sum().reset_index()
compscore = pd.cut(dist_haz['flood-hazard'],bins = 5,precision = 0,labels = compositescorelabels )
dist_haz['flood-hazard'] = compscore
dist_haz = dist_haz.merge(dist_ids, on='district')

topsis['risk-score'] = topsis['risk-score'].astype(int)
dist_risk = topsis.groupby(['district','timeperiod'])['risk-score'].sum().reset_index()
compscore = pd.cut(dist_risk['risk-score'],bins = 5,precision = 0,labels = compositescorelabels )
dist_risk['risk-score'] = compscore
dist_risk = dist_risk.merge(dist_ids, on='district')

# === DISTRICT LEVEL SCORES (improved, minimal changes) ===

# dist_ids = pd.read_csv(os.getcwd()+r'/RiskScoreModel/assets/district_objectid.csv')

# def safe_qcut(s: pd.Series, q: int = 5, labels=None):
#     """
#     Quantile-bin a series but gracefully handle too-few-unique values and duplicates.
#     Ensures number of labels matches number of bins.
#     """
#     s = pd.to_numeric(s, errors='coerce')
#     s_nonan = s.dropna()
#
#     # If everything is NaN or constant -> single bin
#     nunique = s_nonan.nunique()
#     if nunique == 0:
#         return pd.Series(pd.NA, index=s.index, dtype="Int64")
#     if nunique == 1:
#         # All same value -> put everything in bin 1
#         return pd.Series(1, index=s.index, dtype="Int64")
#
#     # Don’t ask for more quantiles than distinct values
#     q_eff = min(q, nunique)
#
#     # First try: direct qcut with adjusted q and labels
#     lab = labels if (labels is not None and len(labels) == q_eff) else list(range(1, q_eff + 1))
#     try:
#         return pd.qcut(s, q=q_eff, labels=lab, duplicates='drop')
#     except ValueError:
#         # If duplicates still collapse bins, get actual bins, then cut with matching labels
#         _, bins = pd.qcut(s, q=q_eff, labels=None, retbins=True, duplicates='drop')
#         nb = len(bins) - 1
#         lab2 = labels if (labels is not None and len(labels) == nb) else list(range(1, nb + 1))
#         return pd.cut(s, bins=bins, labels=lab2, include_lowest=True) # type: ignore
#
# # Use integer labels for easier downstream use
# compositescorelabels = [1, 2, 3, 4, 5]
#
# # Work only with tehsil rows for aggregation (prevents mixing pre-aggregated rows)
# _base = topsis.loc[topsis['tehsil'].notna()].copy()
#
# # Ensure numerics
# for c in ['vulnerability', 'exposure', 'government-response', 'flood-hazard', 'risk-score', 'sum-population']:
#     _base[c] = pd.to_numeric(_base[c], errors='coerce')
#
# # If higher government-response means better capacity (i.e., lowers risk), invert it here for risk aggregation
# #_base['gov_resp_cost'] = 1 - _base['government-response']
#
# def _wavg(g, col, w='sum-population'):
#     v = pd.to_numeric(g[col], errors='coerce')
#     wv = pd.to_numeric(g[w], errors='coerce')
#     m = v.notna() & wv.notna() & (wv > 0)
#     return np.average(v[m], weights=wv[m]) if m.any() else np.nan
#
# # Vulnerability
# dist_vul = (
#     _base.groupby(['district','timeperiod'])
#          .apply(lambda g: pd.Series({'vulnerability': _wavg(g, 'vulnerability')}))
#          .reset_index()
# )
# dist_vul['vulnerability'] = safe_qcut(dist_vul['vulnerability'], q=5, labels=compositescorelabels)
# dist_vul = dist_vul.merge(dist_ids, on='district', how='left')
#
# # Exposure
# dist_exp = (
#     _base.groupby(['district','timeperiod'])
#          .apply(lambda g: pd.Series({'exposure': _wavg(g, 'exposure')}))
#          .reset_index()
# )
# dist_exp['exposure'] = safe_qcut(dist_exp['exposure'], q=5, labels=compositescorelabels)
# dist_exp = dist_exp.merge(dist_ids, on='district', how='left')
#
# # Government response (cost: higher = worse for risk)
# dist_govt = (
#     _base.groupby(['district','timeperiod'])
#          .apply(lambda g: pd.Series({'government-response': _wavg(g, 'government-response')}))
#          .reset_index()
# )
# dist_govt['government-response'] = safe_qcut(dist_govt['government-response'], q=5, labels=compositescorelabels)
# dist_govt = dist_govt.merge(dist_ids, on='district', how='left')
#
# # Flood hazard
# dist_haz = (
#     _base.groupby(['district','timeperiod'])
#          .apply(lambda g: pd.Series({'flood-hazard': _wavg(g, 'flood-hazard')}))
#          .reset_index()
# )
# dist_haz['flood-hazard'] = safe_qcut(dist_haz['flood-hazard'], q=5, labels=compositescorelabels)
# dist_haz = dist_haz.merge(dist_ids, on='district', how='left')
#
# # Risk-score (keep continuous when averaging; bin only after)
# dist_risk = (
#     _base.groupby(['district','timeperiod'])
#          .apply(lambda g: pd.Series({'risk-score': _wavg(g, 'risk-score')}))
#          .reset_index()
# )
# dist_risk['risk-score'] = safe_qcut(dist_risk['risk-score'], q=5, labels=compositescorelabels)
# dist_risk = dist_risk.merge(dist_ids, on='district', how='left')



indicators = ['total-tender-awarded-value', 
    'restoration-measures-tenders-awarded-value',
    'lwss-tenders-awarded-value', 
    'ndrf-tenders-awarded-value', 
    'sdmf-tenders-awarded-value', 
    'wss-tenders-awarded-value', 
    'preparedness-measures-tenders-awarded-value', 
    'immediate-measures-tenders-awarded-value', 
    'others-tenders-awarded-value',
    'relief-and-mitigation-sanction-value',
    #'total-animal-washed-away',
    #'total-animal-affected',
    #'total-house-fully-damaged',
    #'embankments-affected',
    #'roads',
    #'bridge',
    #'embankment-breached',
    "total-livestock-loss",
    "schools-damaged",
    "person-dead",
    "person-major-injury",
    "structure-lost",
    "health-centres-lost",
    "roadlength",
    'sum-population',
    "nviall-comp",
    "sviall-comp",
    "pviall-comp",
    "hviall-comp",
    "fviall-comp",
    "cviall-comp",
    #'inundation-intensity-sum',
    'total-hhd',
    #'human-live-lost',
    'sum-aged-population',
    'schools-count',
    #'HealthCenters',
    'road-length',
    'rail-length',
    'block-nosanitation-hhds-pct',
    'drainage-density',
    #'flood-hazard',
    'inundation-pct',
    'inundation-intensity-mean',
    'inundation-intensity-mean-nonzero',
    'avg-electricity',
    'block-piped-hhds-pct',
    'mean-sex-ratio',
    #'population-affected-total',
    #'crop-area',
    'elevation-mean',
    #'mean-ndvi',
    #'mean-ndbi',
    #'block-area',
    #'are-new',
    #'riverlevel-mean',
    #'riverlevel-min',
    #'riverlevel-max',
    'sum-young-population',
    #'mean-cn',
    'slope-mean',
    'avg-tele',
    'distance-from-river-mean',
    #'water',
    #'trees',
    #'rangeland',
    #'crops',
    #'flooded-vegetation',
    #'built-area',
    #'bare-ground',
    #'clouds',
    'net-sown-area-in-hac',
    'road-count',
    'rail-count',
    'max-rain',
    'mean-rain',
    'sum-rain',
    #'efficiency',

    'mean-daily-runoff',
    'sum-runoff',
    'peak-runoff',
    'topsis-score',
    #'risk-score',
    #'exposure',
    #'vulnerability',
    #'government-response',
    ]

# Define aggregation rules based on the columns
aggregation_rules = {
    # Sum columns
    'total-tender-awarded-value': 'sum', 
    'restoration-measures-tenders-awarded-value': 'sum',
    'lwss-tenders-awarded-value': 'sum', 
    'ndrf-tenders-awarded-value': 'sum', 
    'sdmf-tenders-awarded-value': 'sum', 
    'wss-tenders-awarded-value': 'sum', 
    'preparedness-measures-tenders-awarded-value': 'sum', 
    'immediate-measures-tenders-awarded-value': 'sum', 
    'others-tenders-awarded-value': 'sum',
    'relief-and-mitigation-sanction-value': 'sum',

    'total-tender-awarded-value-fy-cumsum':'sum', 
    'restoration-measures-tenders-awarded-value-fy-cumsum':'sum',
    'lwss-tenders-awarded-value-fy-cumsum':'sum', 
    'ndrf-tenders-awarded-value-fy-cumsum':'sum', 
    'sdmf-tenders-awarded-value-fy-cumsum':'sum', 
    'wss-tenders-awarded-value-fy-cumsum':'sum', 
    'preparedness-measures-tenders-awarded-value-fy-cumsum':'sum', 
    'immediate-measures-tenders-awarded-value-fy-cumsum':'sum', 
    'others-tenders-awarded-value-fy-cumsum':'sum',
    'relief-and-mitigation-sanction-value-fy-cumsum':'sum',

    "total-livestock-loss" : 'sum',
    "schools-damaged": 'sum',
    "person-dead": 'sum',
    "person-major-injury": 'sum',
    "structure-lost": 'sum',
    "health-centres-lost": 'sum',
    "roadlength": 'sum',

    'sum-population': 'sum',
    'inundation-intensity-sum': 'sum',
    'total-hhd': 'sum',
    'sum-aged-population': 'sum',
    'schools-count': 'sum',
    #'healthcenters': 'sum',
    'road-length': 'sum',
    'rail-length': 'sum',
    'sum-rain': 'sum',
    #'block-area':'sum',
    'sum-young-population':'sum',
    'net-sown-area-in-hac':'sum',
    'road-count':'sum',
    'rail-count':'sum',

    # Mean for percentage or density-based metrics
    'block-nosanitation-hhds-pct': 'mean',
    'drainage-density': 'mean',
    'inundation-pct': 'mean',
    'inundation-intensity-mean-nonzero': 'mean',
    'inundation-intensity-mean': 'mean',
    'avg-electricity': 'mean',
    'block-piped-hhds-pct': 'mean',
    'mean-sex-ratio': 'mean',
    'mean-rain':'mean',
    'elevation-mean':'mean',
    'slope-mean':'mean',
    'avg-tele':'mean',
    'distance-from-river-mean':'mean',
    'mean-daily-runoff':'mean',
    'sum-runoff':'sum',
    'peak-runoff':'max',

    #'efficiency':'mean',
    "nviall-comp":'mean',
    "sviall-comp":'mean',
    "pviall-comp":'mean',
    "hviall-comp":'mean',
    "fviall-comp":'mean',
    "cviall-comp":'mean',

    'topsis-score': 'mean',
    #'risk-score': 'mean',
    #'exposure': 'mean',
    #'vulnerability': 'mean',
    #'government-response': 'mean',
    #'flood-hazard': 'mean',

    # Max for hazard levels
    
    'max-rain':'max',

}

rounding_rules = {

    'total-tender-awarded-value':0, 
    'restoration-measures-tenders-awarded-value':0,
    'lwss-tenders-awarded-value':0, 
    'ndrf-tenders-awarded-value':0, 
    'sdmf-tenders-awarded-value':0, 
    'wss-tenders-awarded-value':0, 
    'preparedness-measures-tenders-awarded-value':0, 
    'immediate-measures-tenders-awarded-value':0, 
    'others-tenders-awarded-value':0,
    'relief-and-mitigation-sanction-value':0,
    'net-sown-area-in-hac':0,

    'avg-tele': 1,  # Round column 'A' to 1 decimal place
    'avg-electricity': 1,

    'mean-sex-ratio': 2,  
    'inundation-intensity-mean-nonzero': 2,  
    'block-piped-hhds-pct':2,
    'block-nosanitation-hhds-pct':2,
    'inundation-intensity-sum':2,
    'max-rain':2,
    'mean-rain':2,
    'sum-rain':2,
    'mean-daily-runoff':2,
    'sum-runoff':2,
    'peak-runoff':2,

    "nviall-comp":2,
    "sviall-comp":2,
    "pviall-comp":2,
    "hviall-comp":2,
    "fviall-comp":2,
    "cviall-comp":2,
    
    'sum-aged-population': 0,   # Round column 'C' to no decimal places
    'sum-young-population': 0,
    'sum-population':0,
    'rail-length':0,
    'road-length':0,
    'elevation-mean':0,
    'slope-mean':0,
    'total-hhd': 0,
    #'crop-area':0,

    #'flood-hazard':0,
    #'risk-score': 0,
    #'exposure': 0,
    #'vulnerability': 0,
    #'government-response': 0,
}

dist_indicators = topsis.groupby(['district', 'timeperiod']).agg(aggregation_rules).reset_index()
dist_indicators = dist_indicators.merge(dist_ids, on='district')


def apply_rounding_rules(df, rounding_rules):

    for column, decimals in rounding_rules.items():
        if column in df.columns:
            df[column] = df[column].round(decimals)
        else:
            print(f"Column {column} does not exist in DataFrame.")
    return df


dist = pd.concat([dist_vul.set_index(['district', 'timeperiod']),#['vulnerability'],
                  dist_exp.set_index(['district', 'timeperiod'])['exposure'],
                  dist_govt.set_index(['district', 'timeperiod'])['government-response'],
                  dist_haz.set_index(['district', 'timeperiod'])['flood-hazard'],
                  dist_risk.set_index(['district', 'timeperiod'])['risk-score'],
                  dist_indicators.set_index(['district', 'timeperiod'])[indicators]],
                  axis=1).reset_index()

#for debugging
#dist.to_csv(os.getcwd()+r'/RiskScoreModel/data/dist_test.csv')
#topsis.to_csv(os.getcwd()+r'/RiskScoreModel/data/topsis_test.csv')
#print(topsis.shape)

final = pd.concat([topsis, dist], ignore_index=True)

# Apply rounding rules
final = apply_rounding_rules(final, rounding_rules)
#final['inundation-pct'] = final['inundation-pct']*100

final = final.rename(columns={"nviall-comp":"natural-vulnerability-index",
    "sviall-comp":"social-vulnerability-index",
    "pviall-comp":"physical-vulnerability-index",
    "hviall-comp":"human-vulnerability-index",
    "fviall-comp":"financial-vulnerability-index",
    "cviall-comp":"composite-vulnerability-index",
    'block-piped-hhds-pct': 'tehsil-piped-hhds-pct',
    'block-nosanitation-hhds-pct': 'tehsil-nosanitation-hhds-pct',
    'mean-sexratio':'sexratio'})

# Add financial year details at the district level as well
final['financial-year'] = final['timeperiod'].apply(lambda x: get_financial_year(x))

final = final.drop(columns=['objectid', 'object-id-new','timeperiod-datetime','year','unnamed:-0'])#,'Unnamed: 0'])

final["total-infrastructure-damage"] =  final["structure-lost"] + final["health-centres-lost"] + final["schools-damaged"]
final.to_csv(os.getcwd()+r'/RiskScoreModel/data/risk_score_final_district.csv', index=False)
