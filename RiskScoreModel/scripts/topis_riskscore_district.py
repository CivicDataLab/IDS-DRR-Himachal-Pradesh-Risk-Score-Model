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
#
# A district score is a roll-up of its tehsil scores. Summing the 1-5 ordinal
# labels (the old behaviour) makes the district score a proxy for *how many
# tehsils the district has*: KANGRA has 22 tehsils and LAHAUL & SPITI has 2, so
# the ceiling for KANGRA was 110 and for LAHAUL & SPITI 10 - against bins cut
# over the pooled range, LAHAUL & SPITI and KINNAUR could never leave class 1
# and small-but-exposed districts like BILASPUR, HAMIRPUR and UNA were pinned
# at 2. Spearman rho between the district risk class and the tehsil count was
# 0.90 under summing.
#
# Assam and Bihar both use an unweighted .mean() instead, which removes that
# artefact but treats a 3.5k-person tehsil and a 307k-person one alike. We take
# the population-weighted mean, so a district score answers "what risk level
# does the average resident of this district face?".
#
# The published district 'topsis-score' is weighted the same way, so the
# continuous column and the 1-5 class always agree. See DIST_SCALE below for
# how the weighted mean becomes that class.
dist_ids = pd.read_csv(os.getcwd()+r'/RiskScoreModel/assets/district_objectid.csv')

compositescorelabels = ['1','2','3','4','5']

# Weight column for the roll-up, and the aggregation used for flood-hazard.
# Hazard is a physical quantity rather than a people-facing one, so set this to
# 'max' if a district should take the worst hazard among its tehsils instead of
# the population-weighted average.
DIST_WEIGHT = 'sum-population'
HAZARD_AGG = 'wmean'   # 'wmean' | 'max'

# How the district value becomes a 1-5 class.
#   'round' - round the weighted mean to the nearest integer (default)
#   'qcut'  - quantile bins over the pooled district x month values
#   'cut'   - equal-width bins over the pooled range (the old behaviour)
#
# The tehsil factors are already on a 1-5 scale, so their weighted mean is
# already in 1-5 and rounding keeps it there: a district hazard of 4 means the
# average resident lives in a hazard-4 tehsil, which is directly comparable to
# the tehsil column and to the same district in another month.
#
# Re-binning does not survive HP's geography. Only 12 districts contribute and
# their values are bimodal - KINNAUR and LAHAUL & SPITI sit near 1 while the
# rest cluster high - so equal-width bins put every cut-point inside that gap:
# exposure came out as 132 district-months in class 1, 330 in class 4, 330 in
# class 5 and *nobody* in classes 2 or 3. Quantile bins fill all five classes
# but make the number relative to HP's own history rather than absolute, so a
# weighted-mean hazard of 4.0 lands in class 3 because that is HP's median.
DIST_SCALE = 'round'  # 'round' | 'qcut' | 'cut'

# Only tehsil rows feed the roll-up; district rows are appended further down.
tehsil_rows = topsis[topsis['tehsil'].notna()].copy()
topsis['risk-score'] = topsis['risk-score'].astype(int)
tehsil_rows['risk-score'] = tehsil_rows['risk-score'].astype(int)


def _weighted_mean(frame, col, weight=DIST_WEIGHT):
    values = pd.to_numeric(frame[col], errors='coerce')
    weights = pd.to_numeric(frame[weight], errors='coerce')
    usable = values.notna() & weights.notna() & (weights > 0)
    if not usable.any():
        # No population to weight by: fall back to the plain mean rather than
        # dropping the district out of the output entirely.
        return values.mean()
    return np.average(values[usable], weights=weights[usable])


def _to_class_1_to_5(series, native_scale=True):
    """Turn a district-level value into the published 1-5 class.

    native_scale says the value is already a 1-5 mean (the four factors) and so
    can simply be rounded; the TOPSIS score is continuous and always needs bins.
    """
    if DIST_SCALE == 'round' and native_scale:
        return series.round().clip(1, 5).astype(int).astype(str)
    if DIST_SCALE == 'cut':
        return pd.cut(series, bins=5, precision=0, labels=compositescorelabels)
    # duplicates='drop' because the ordinal factors tie heavily; when bins
    # collapse, qcut needs the label count to match the bins that survive.
    _, bins = pd.qcut(series, q=5, retbins=True, duplicates='drop')
    labels = compositescorelabels[:len(bins) - 1]
    return pd.cut(series, bins=bins, labels=labels, include_lowest=True)


def rollup_to_district(col, how='wmean', native_scale=True):
    """Roll a tehsil column up to district x month, then class it 1-5."""
    if how == 'wmean':
        agg = (tehsil_rows.groupby(['district', 'timeperiod'])
                          .apply(lambda g: _weighted_mean(g, col), include_groups=False)
                          .reset_index(name=col))
    else:
        agg = tehsil_rows.groupby(['district', 'timeperiod'])[col].agg(how).reset_index()
    agg[col] = _to_class_1_to_5(agg[col], native_scale=native_scale)
    return agg.merge(dist_ids, on='district')


dist_vul = rollup_to_district('vulnerability')
dist_exp = rollup_to_district('exposure')
dist_govt = rollup_to_district('government-response')
dist_haz = rollup_to_district('flood-hazard', how=HAZARD_AGG)

# Risk rolls up from the 1-5 'risk-score' under DIST_SCALE='round', so that the
# district number stays on the same scale as the tehsil one. Under 'qcut'/'cut'
# it rolls up from the continuous 'topsis-score' instead, which keeps the
# magnitude the weighting acts on rather than binning an already-binned label.
if DIST_SCALE == 'round':
    dist_risk = rollup_to_district('risk-score')
else:
    dist_risk = (rollup_to_district('topsis-score', native_scale=False)
                 .rename(columns={'topsis-score': 'risk-score'}))



indicators = ['total-tender-awarded-value', 
    'sdrf-tenders-awarded-value',
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
    'sdrf-tenders-awarded-value': 'sum',
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
    'sdrf-tenders-awarded-value':0,
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

dist_indicators = tehsil_rows.groupby(['district', 'timeperiod']).agg(aggregation_rules).reset_index()

# The published district 'topsis-score' has to be the same number the district
# 'risk-score' class was cut from, so weight it the same way instead of taking
# the plain mean the aggregation_rules above would give.
dist_indicators = dist_indicators.drop(columns=['topsis-score']).merge(
    (tehsil_rows.groupby(['district', 'timeperiod'])
                .apply(lambda g: _weighted_mean(g, 'topsis-score'), include_groups=False)
                .reset_index(name='topsis-score')),
    on=['district', 'timeperiod'])

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
