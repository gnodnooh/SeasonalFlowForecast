#!/Users/dlee/anaconda3/envs/dlee/bin/python
# -*- coding: utf-8 -*-
"""
This script calculates performances of GranD Dam inflow predictions
and classify dams with dam characteristics analyzed by Jia.

Donghoon Lee @ Jul-9-2019
dlee298@wisc.edu
"""
import os
os.environ['R_HOME'] = '/Users/dlee/anaconda3/envs/dlee/lib/R'  # The R path can be found by $ R RHOME
from rpy2.robjects import r
from rpy2.robjects import pandas2ri
import rpy2.robjects as robjects
pandas2ri.activate()
import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd
import metrics as mt
import HydroErr as he
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import pysal as ps
import time
import seaborn as sns
from scipy import stats

# Load monthly GranD dam inflow from SUTD
dfFlowDams = pd.read_hdf('./data/dfFlowDams.hdf')
ind_dams = np.load('./data/ind_dams.npz')['ind_dams']
damList = dfFlowDams.columns.values
# - Define High-flow Season
clim = dfFlowDams.groupby(dfFlowDams.index.month).mean()
clim_prct = clim/clim.sum(axis=0)
clim3 = dfFlowDams.rolling(3, min_periods=1, center=True).mean()
clim3 = clim3.groupby(dfFlowDams.index.month).mean()
pm3 = pd.concat([clim3.idxmax()-1, clim3.idxmax(), clim3.idxmax()+1], axis=1)
pm3[pm3 > 12] = pm3[pm3 > 12] - 12
pm3[pm3 < 1] = pm3[pm3 < 1] + 12

# Load monthly predictions (time-series and signs of "isFcst")
dfMP1 = pd.read_hdf('./prediction/dfMP1.hdf')
dfMP1_sign = pd.read_hdf('./prediction/dfMP1_sign.hdf')
dfMP2 = pd.read_hdf('./prediction/dfMP2.hdf')
dfMP2_sign = pd.read_hdf('./prediction/dfMP2_sign.hdf')
dfMP3 = pd.read_hdf('./prediction/dfMP3.hdf')
dfMP3_sign = pd.read_hdf('./prediction/dfMP3_sign.hdf')
temp = pd.DataFrame(np.ones([1, damList.shape[0]]), index=np.array([13]), columns=damList)
dfMP1_sign = dfMP1_sign.append(temp)
dfMP2_sign = dfMP2_sign.append(temp)
dfMP3_sign = dfMP3_sign.append(temp)

# Load hydropower production results
dfProd = pd.read_hdf('./hydropower_result/hydropower_production.hdf')
# - Improvement is defined as (MPC - SDP_t)/MPC_PF
impv_df = (dfProd.loc['mpc_df3'] - dfProd.loc['sdp_t'])/dfProd.loc['mpc_pf']*100
impv_pf = (dfProd.loc['mpc_pf'] - dfProd.loc['sdp_t'])/dfProd.loc['mpc_pf']*100
# =============================================================================
# impv_df = (dfProd.loc['mpc_df3'] - dfProd.loc['sdp_t'])/dfProd.loc['sdp_t']*100
# impv_pf = (dfProd.loc['mpc_pf'] - dfProd.loc['sdp_t'])/dfProd.loc['sdp_t']*100
# =============================================================================
impv_df.name = 'impv_df'
impv_pf.name = 'impv_pf'

# Load 1593 GranD Dam shapefile
# TODO: The index is not sorted
gdfDam = gpd.read_file('./data/granddams_eval.shp')
gdfDam = gdfDam.drop(gdfDam.columns[1:-1], axis=1)

# Load Jia's results
fitted = pd.read_csv('./hydropower_result/fitted_values.csv')
fitted.index.name = 'GRAND_ID'
corr_nse = pd.read_csv('./hydropower_result/correlation.csv')
corr_nse.index = fitted.index

# DOR values of 1593 GranD dams
df_dor = pd.read_hdf('./data/df_dor.hdf')

# Regression variables
df_var = pd.read_hdf('./hydropower_result/regression_variables.hdf')


#%% Calculate performances of prediction
'''
- Nash-Sutcliffe Efficiency (NSE):              -inf < NSE ≤ 1
- Modified Nash-Sutcliffe Efficiency (NSE_mod): -inf < MNSE ≤ 1
- Kling-Gupta efficiency (2012) (KGE):          -inf < KGE ≤ 1
- Heidke Skill Score (HSS):                     -inf < HSS ≤ 1
- Peirce Skill Score (PSS):                     -1 ≤ PSS ≤ 1
- Gerrity Skill Score (GSS):                    -1 ≤ GSS ≤ 1
'''
fn_score = './data/PredSkillScores.hdf'
if not os.path.exists(fn_score):
    stime = time.time()
    scoreList = ['nse', 'nse_mod', 'kge', 'hss', 'pss', 'gss']
    leadList = ['t1', 't2', 't3']
    strmon = [x for x in range(1,14)]
    iterables = [scoreList, leadList, strmon]
    multiIndex = pd.MultiIndex.from_product(iterables, names=['index','lead','month'])
    data = np.full([len(scoreList)*3*13, damList.shape[0]], np.nan)
    score = pd.DataFrame(data, multiIndex, damList)
    for did in damList:
        obs = dfFlowDams[did]
        sim1 = dfMP1[did]
        sim2 = dfMP2[did]
        sim3 = dfMP3[did]
        for m in range(1,14):
            if m != 13:
                mdx = obs.index[obs.index.month == m]
            else:
                mdx = obs.index
            # MP1 model
            # Deterministic Scores
            score.loc[('nse','t1',m)][did] = he.nse(sim1[mdx], obs[mdx])    # NSE
            score.loc[('nse_mod','t1',m)][did] = he.nse_mod(sim1[mdx], obs[mdx])    # NSE
            score.loc[('kge','t1',m)][did] = he.kge_2012(sim1[mdx], obs[mdx])       # KGE_2012
            # Multicategorical Scores
            table = mt.makeContTableANB(obs[mdx], sim1[mdx])
            mct = mt.MulticlassContingencyTable(table, n_classes=3)
            score.loc[('hss','t1',m)][did] = mct.heidke_skill_score()       # HSS
            score.loc[('pss','t1',m)][did] = mct.peirce_skill_score()       # PSS
            score.loc[('gss','t1',m)][did] = mct.gerrity_skill_score()      # GSS
            # MP2 model
            # Deterministic Scores
            score.loc[('nse','t2',m)][did] = he.nse(sim2[mdx], obs[mdx])    # NSE
            score.loc[('nse_mod','t2',m)][did] = he.nse_mod(sim2[mdx], obs[mdx])    # NSE
            score.loc[('kge','t2',m)][did] = he.kge_2012(sim2[mdx], obs[mdx])       # KGE_2012
            # Multicategorical Scores
            table = mt.makeContTableANB(obs[mdx], sim2[mdx])
            mct = mt.MulticlassContingencyTable(table, n_classes=3)
            score.loc[('hss','t2',m)][did] = mct.heidke_skill_score()       # HSS
            score.loc[('pss','t2',m)][did] = mct.peirce_skill_score()       # PSS
            score.loc[('gss','t2',m)][did] = mct.gerrity_skill_score()      # GSS
            # MP3 model
            # Deterministic Scores
            score.loc[('nse','t3',m)][did] = he.nse(sim3[mdx], obs[mdx])    # NSE
            score.loc[('nse_mod','t3',m)][did] = he.nse_mod(sim3[mdx], obs[mdx])    # NSE
            score.loc[('kge','t3',m)][did] = he.kge_2012(sim3[mdx], obs[mdx])       # KGE_2012
            # Multicategorical Scores
            table = mt.makeContTableANB(obs[mdx], sim3[mdx])
            mct = mt.MulticlassContingencyTable(table, n_classes=3)
            score.loc[('hss','t3',m)][did] = mct.heidke_skill_score()       # HSS
            score.loc[('pss','t3',m)][did] = mct.peirce_skill_score()       # PSS
            score.loc[('gss','t3',m)][did] = mct.gerrity_skill_score()      # GSS

    # TODO: Replace subset of multiindexframe: case of KGE          
    score[score < -0.4] = 0
    score[score.isnull()] = 0
    score = score.sort_index()
    # Save as HDF format
    score.to_hdf(fn_score, key='df', complib='blosc:zstd', complevel=9)
    print('%s is saved.' % fn_score)
    print('%.3fs is taken.' % (time.time() - stime))
else:
    score = pd.read_hdf(fn_score)
    print('%s is loaded.' % fn_score)


#%% Performance indices of Prediction
fn_indices = './data/PredIndices.hdf'
if not os.path.exists(fn_indices):
    nse1 = score.loc[('nse','t1',1)]
    nse12 = score.loc[('nse','t1',13)]
    nse_mod1 = score.loc[('nse_mod','t1',1)]
    nse_mod12 = score.loc[('nse_mod','t1',13)]
    kge1 = score.loc[('kge','t1',1)]
    kge12 = score.loc[('kge','t1',13)]
    hss1 = score.loc[('hss','t1',1)]
    hss12 = score.loc[('hss','t1',13)]
    pss1 = score.loc[('pss','t1',1)]
    pss12 = score.loc[('pss','t1',13)]
    gss1 = score.loc[('gss','t1',1)]
    gss12 = score.loc[('gss','t1',13)]
    # Volumetric Score (VS)
    # The VS is monthly averaged score 
    # VS = monthly scores * ratio of monthly flow volume to annual volume
    nse_vs = np.zeros([damList.shape[0], 2])
    nse_mod_vs = nse_vs.copy()
    kge_vs = nse_vs.copy()
    hss_vs = nse_vs.copy()
    pss_vs = nse_vs.copy()
    gss_vs = nse_vs.copy()
    for (i, did) in enumerate(damList):
        dam_clim_prct = clim_prct[did]
        dam_high_prct = clim_prct[did][pm3.loc[did]]/np.sum(clim_prct[did][pm3.loc[did]])
        nse_vs[i,0] = np.sum(score.loc[('nse','t1')][did][:-1]*dam_clim_prct)
        nse_vs[i,1] = np.sum(score.loc[('nse','t1', pm3.loc[did])][did].values * dam_high_prct)
        nse_mod_vs[i,0] = np.sum(score.loc[('nse_mod','t1')][did][:-1]*dam_clim_prct)
        nse_mod_vs[i,1] = np.sum(score.loc[('nse_mod','t1', pm3.loc[did])][did].values * dam_high_prct)
        kge_vs[i,0] = np.sum(score.loc[('kge','t1')][did][:-1]*dam_clim_prct)
        kge_vs[i,1] = np.sum(score.loc[('kge','t1', pm3.loc[did])][did].values * dam_high_prct)
        hss_vs[i,0] = np.sum(score.loc[('hss','t1')][did][:-1]*dam_clim_prct)
        hss_vs[i,1] = np.sum(score.loc[('hss','t1', pm3.loc[did])][did].values * dam_high_prct)    
        pss_vs[i,0] = np.sum(score.loc[('pss','t1')][did][:-1]*dam_clim_prct)
        pss_vs[i,1] = np.sum(score.loc[('pss','t1', pm3.loc[did])][did].values * dam_high_prct)
        gss_vs[i,0] = np.sum(score.loc[('gss','t1')][did][:-1]*dam_clim_prct)
        gss_vs[i,1] = np.sum(score.loc[('gss','t1', pm3.loc[did])][did].values * dam_high_prct)
    nse_vs12 = pd.Series(nse_vs[:,0], nse1.index)
    nse_vs3 = pd.Series(nse_vs[:,1], nse1.index)
    nse_mod_vs12 = pd.Series(nse_mod_vs[:,0], nse1.index)
    nse_mod_vs3 = pd.Series(nse_mod_vs[:,1], nse1.index)
    kge_vs12 = pd.Series(kge_vs[:,0], nse1.index)
    kge_vs3 = pd.Series(kge_vs[:,1], nse1.index)
    hss_vs12 = pd.Series(hss_vs[:,0], nse1.index)
    hss_vs3 = pd.Series(hss_vs[:,1], nse1.index)
    pss_vs12 = pd.Series(pss_vs[:,0], nse1.index)
    pss_vs3 = pd.Series(pss_vs[:,1], nse1.index)
    gss_vs12 = pd.Series(gss_vs[:,0], nse1.index)
    gss_vs3 = pd.Series(gss_vs[:,1], nse1.index)
    # Dataframe of Indices
    indices = pd.concat([impv_pf, impv_df, fitted,
                         nse12, nse_vs3, nse_vs12,
                         nse_mod12, nse_mod_vs3, nse_mod_vs12,
                         kge12, kge_vs3, kge_vs12,
                         hss12, hss_vs3, hss_vs12,
                         pss12, pss_vs3, pss_vs12,
                         gss12, gss_vs3, gss_vs12
                         ], axis=1)
    indices_name = ['PF', 'DF', 'DAM',
                    'NSE12', 'NSE_VS_HIGH', 'NSE_VS_12',
                    'MNSE12', 'MNSE_VS_HIGH', 'MNSE_VS_12',
                    'KGE12', 'KGE_VS_HIGH', 'KGE_VS_12',
                    'HSS12', 'HSS_VS_HIGH', 'HSS_VS_12',
                    'PSS12', 'PSS_VS_HIGH', 'PSS_VS_12',
                    'GSS12', 'GSS_VS_HIGH', 'GSS_VS_12']
    indices.columns = indices_name
    indices.index.name = 'GRAND_ID'
    indices.to_hdf(fn_indices, key='df', complib='blosc:zstd', complevel=9)
    print('%s is saved.' % fn_indices)
else:
    indices = pd.read_hdf(fn_indices)
    print('%s is loaded.' % fn_indices)

# Reset of Improvement of hydropower production
if True:
    # Upper line is MPC_PF
    impv_df = (dfProd.loc['mpc_df3'] - dfProd.loc['sdp_t'])/dfProd.loc['mpc_pf']*100
    impv_pf = (dfProd.loc['mpc_pf'] - dfProd.loc['sdp_t'])/dfProd.loc['mpc_pf']*100
else:
    # Upper line is SDP
    impv_df = (dfProd.loc['mpc_df3'] - dfProd.loc['sdp_t'])/dfProd.loc['sdp_t']*100
    impv_pf = (dfProd.loc['mpc_pf'] - dfProd.loc['sdp_t'])/dfProd.loc['sdp_t']*100
impv_df.name = 'DF'
impv_pf.name = 'PF'
indices['DF'] = impv_df
indices['PF'] = impv_pf

# Export data for Jia
if False:
    temp = indices[indices.columns[[0,1,2,9,11,18,20]]]
    temp.columns = ['PF', 'DF', 'DAM', 'KGE', 'Weighted KGE', 'GSS', 'Weighted GSS']
    df_all = pd.concat([df_dor['DOR'], temp.sort_index()], axis=1)
    df_all.to_csv('./data/all_indices.csv')


#%% Select dams with relatively lower DOR values
# Sort the order of GRAND_ID
indices = indices.sort_index()
df_dor = df_dor.sort_index()
assert np.all(indices.index == df_dor.index)
if True:
    # Remove dams with DOR values
    # 100% DOR leaves 1,055 dams (66.2%)
    # 90% DOR leaves 1,015 dams (63.7%)
    trsd_dor = 105
    lowDOR = df_dor['DOR'] < trsd_dor
    indicesReduced = indices.iloc[np.where(lowDOR)]
    ndam = indicesReduced.shape[0]
    print('----------------------------------------------')
    print('%d (%.1f%%) dams are removed when DOR < %d' % 
          (1593-ndam, np.round((1593-ndam)/1593*10000, 1)/100, trsd_dor) )
    print('Averaged DOR of %d dams: %.1f%%' % (ndam, df_dor[lowDOR]['DOR'].mean()))
    # How much of the CAP_MCM we lose by DOR?
    caploss = 1 - (df_dor['CAP_MCM'][indicesReduced.index].sum() / df_dor['CAP_MCM'].sum())
    print('Percentage of CAP_MCM excluded: %.1f%%' % (caploss*100))
    # How much of the hydropower production we lose by DOR?
    prodloss = 100 - dfProd[indicesReduced.index].sum(axis=1) / dfProd.sum(axis=1) * 100
    print('Loss of hydropower production:')
    print(prodloss)
    print('----------------------------------------------')


#%% Scatterplots of KGE and GSS
subset = indicesReduced[indicesReduced.columns[[0,1,2,9,11,18,20]]]
subset.columns = ['PF', 'DF', 'DAM', 'KGE', 'Weighted KGE', 'GSS', 'Weighted GSS']
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8.5,8), sharey=True)
sns.set_style('white')

fcst_type = 'DF'

if fcst_type == 'DF':
    cmap = sns.cubehelix_palette(dark=.1, light=.8, as_cmap=True)
else:
    cmap = sns.cubehelix_palette(dark=.1, light=.8, rot=-0.4, as_cmap=True)
for (i, el) in enumerate(subset.columns[3:]):
    ax = axes.flatten('C')[i]
    r_dam = stats.pearsonr(subset[el],subset['DAM'])[0] 
    r_fcst = stats.pearsonr(subset[el],subset[fcst_type])[0]
    sc = sns.scatterplot(x=el, y='DAM', data=subset, ax=ax, marker='o', linewidth=0.5, 
                    alpha=0.5, palette=cmap, hue=fcst_type, 
                    size=fcst_type, sizes=(5,200),
                    hue_norm=(-10, 50), size_norm=(-10,50))
    sc.set_ylabel('DAM fitted value', fontsize=14)
    sc.set_xlabel(el, fontsize=14)
    sc.tick_params(labelsize=12)
    ax.annotate('R_DAM=%.3f, R_IMPV=%.3f'% (r_dam, r_fcst), (-0.05,-0.17))
    ax.set_xlim([-0.1, 1])
    ax.set_ylim([-0.2, 0.6])
    ax.plot([-1, 1], [0.1, 0.1], color='grey', lw=1, alpha=0.5, linestyle='--')
    if i == 1:
        xt = 0.3
        ax.plot([xt, xt], [-1, 1], color='grey', lw=1, alpha=0.5, linestyle='--')
    elif i == 3:
        xt = 0.3
        ax.plot([xt, xt], [-1, 1], color='grey', lw=1, alpha=0.5, linestyle='--')
    if i == 1:
        ax.legend(bbox_to_anchor=(1.01, 1.025), loc=2)
    else:
        ax.get_legend().remove()

plt.tight_layout()
plt.show()
fn_save = './figures/Multiplots_KGE_GSS_%s.png' % fcst_type
fig.savefig(fn_save)
print('%s is saved.' % fn_save)

# Correlation between indices
indices_corr = subset.corr()
print(indices_corr)




#%% Boxplots of Dam, Forecast, Dam & Forecast
# Dam classification
good_dam = subset['DAM'] > 0.1              # Fitted model > 0.1
good_for = subset['Weighted KGE'] > 0.3     # KGE
indpp = ~good_dam & ~good_for       # Poor-Dam & Poor-Forecast
indgp = good_dam & ~good_for        # Good-Dam & Poor-Forecast
indpg = ~good_dam & good_for        # Poot-Dam & Good-Forecast
indgg = good_dam & good_for         # Good-Dam & Good-Forecast
impv2D = pd.concat([subset[ind]['DF'] for ind in list((~good_dam, good_dam))], axis=1)
impv2D.columns = ['PD', 'GD']
impv2F = pd.concat([subset[ind]['DF'] for ind in list((~good_for, good_for))], axis=1)
impv2F.columns = ['PF', 'GD']
impv4 = pd.concat([subset[ind]['DF'] for ind in list((indpp, indgp, indpg, indgg))], axis=1)
impv4.columns = ['PP', 'GP', 'PG', 'GG']


# Boxplot of improvments in hydropower production
fsize, lsize = 15, 13.5
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10,5), sharey=True,
                         gridspec_kw={'width_ratios': [1,1,2]})
sns.set_style('white')
# a) Only Dams
ax = axes[0]
sns.boxplot(data=impv2D, ax=ax, width=0.7, palette='deep')
ax.set_ylim([-20,40])
ax.set_ylabel('Improvement in hydropower production (%)', fontsize=fsize)
ax.tick_params(labelsize=lsize)
text_xl = ['Poor-Dam\n(%.1f%%)' % (np.round(np.sum(~good_dam)/ndam,4)*100), 
           'Good-Dam\n(%.1f%%)' % (np.round(np.sum(good_dam)/ndam,4)*100)]
xl = ax.set_xticklabels(text_xl, fontsize=lsize)
# b) Only Forecasts
ax = axes[1]
sns.boxplot(data=impv2F, ax=ax, width=0.7, palette='deep')
ax.set_ylim([-20,40])
ax.tick_params(labelsize=lsize)
text_xl = ['Poor-Fcst\n(%.1f%%)' % (np.round(np.sum(~good_for)/ndam,4)*100), 
           'Good-Fcst\n(%.1f%%)' % (np.round(np.sum(good_for)/ndam,4)*100)]
xl = ax.set_xticklabels(text_xl, fontsize=lsize)
# c) Dams & Forecasts
ax = axes[2]
sns.boxplot(data=impv4, ax=ax, width=0.7, palette='deep')
ax.set_ylim([-20,40])
ax.tick_params(labelsize=lsize)
text_xl = ['Poor-Dam\nPoor-Fcst\n(%.1f%%)' % (np.round(np.sum(indpp)/ndam,4)*100), 
           'Good-Dam\nPoor-Fcst\n(%.1f%%)' % (np.round(np.sum(indgp)/ndam,4)*100), 
           'Poor-Dam\nGood-Fcst\n(%.1f%%)' % (np.round(np.sum(indpg)/ndam,4)*100), 
           'Good-Dam\nGood-Fcst\n(%.1f%%)' % (np.round(np.sum(indgg)/ndam,4)*100)]
xl = ax.set_xticklabels(text_xl, fontsize=lsize)
plt.tight_layout(w_pad=0.5)
plt.show()

# Print improvements in each category
impv2D_mean = impv2D.mean().values
impv2F_mean = impv2F.mean().values
impv4_mean = impv4.mean().values
print('---------------------------------------------')
print('Averaged improvements in each dam category')
print('Poor-Dam: %.1f%%' % impv2D_mean[0])
print('Good-Dam: %.1f%%' % impv2D_mean[1])
print('Poor-Fcst: %.1f%%' % impv2F_mean[0])
print('Good-Fcst: %.1f%%' % impv2F_mean[1])
print('Poor-Dam & Poor-Fcst: %.1f%%' % impv4_mean[0])
print('Good-Dam & Poor-Fcst: %.1f%%' % impv4_mean[1])
print('Poor-Dam & Good-Fcst: %.1f%%' % impv4_mean[2])
print('Good-Dam & Good-Fcst: %.1f%%' % impv4_mean[3])
print('---------------------------------------------')

#fig.savefig('./figures/boxplot_all.png')



#%%
if False:
    # Boxplot of improvments of hydropower production
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,5))
    sns.set_style('white')
    sns.boxplot(data=impv4, ax=ax, width=0.7, palette='deep')
    #ax.set_ylim([-0.2, 0.6])
    ax.set_ylim([-20,40])
    ax.set_ylabel('Improvement in hydropower production (%)', fontsize=14)
    ax.tick_params(labelsize=14)
    text_xl = ['Poor-Dam\nPoor-Forecast\n(%.1f%%)' % (np.round(np.sum(indpp)/ndam,4)*100), 
               'Good-Dam\nPoor-Forecast\n(%.1f%%)' % (np.round(np.sum(indgp)/ndam,4)*100), 
               'Poor-Dam\nGood-Forecast\n(%.1f%%)' % (np.round(np.sum(indpg)/ndam,4)*100), 
               'Good-Dam\nGood-Forecast\n(%.1f%%)' % (np.round(np.sum(indgg)/ndam,4)*100)]
    xl = ax.set_xticklabels(text_xl, fontsize=14)
    plt.tight_layout()
    plt.show()
    print(impv4.mean())






#%% Improvements in PF and DF
if False:
    cmap = plt.cm.get_cmap('RdYlGn', 10)
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    vmin, vmax = -25, 25
    
    
    # =============================================================================
    # # Import data into GeoDataFrame
    # gdfDam = gdfDam.merge(impv_df, on='GRAND_ID')
    # gdfDam = gdfDam.merge(impv_pf, on='GRAND_ID')
    # =============================================================================
    
    
    fig, ax = plt.subplots(figsize=(10,4))
    world.plot(ax=ax, color='white', edgecolor='black')
    ax.set_axis_off()
    gdfDam.plot(ax=ax, column='impv_pf', cmap=cmap, markersize=25, edgecolor='black',
                vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    cbar = fig.colorbar(sm)
    plt.axis('equal')
    plt.show()



#%% Improvements in 4 classification
if False:
    cmap = plt.cm.get_cmap('RdYlGn', 10)
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    vmin, vmax = -25, 25
    
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(9,6))
    for (i, el) in enumerate(indList):
        ax = axes.flatten('F')[i]
        world.plot(ax=ax, color='white', edgecolor='black')
        ax.set_axis_off()
        ax.set_title(indName[i])
        selected = gdfDam.loc[gdfDam['GRAND_ID'].isin(indList[i].index[indList[i]])]
        selected.plot(ax=ax, column='impv_df', cmap=cmap, markersize=25, 
                    edgecolor='black', vmin=vmin, vmax=vmax)
    #sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    #sm._A = []
    #cbar = fig.colorbar(sm)
    plt.tight_layout()
    plt.axis('equal')
    plt.show()




#%% Old figures
# =============================================================================
# #%% Scatterplots of all indices
# # DF
# fcst_type = 'DF'
# cmap = sns.cubehelix_palette(dark=.1, light=.8, as_cmap=True)
# fig, axes = plt.subplots(nrows=(indices.columns.shape[0]-3)//3, ncols=3, 
#                          figsize=(12,24), sharey=False)
# sns.set_style('white')
# for (i, el) in enumerate(indices.columns[3:]):
#     ax = axes.flatten('C')[i]
#     r_dam = stats.pearsonr(indices[el],indices['DAM'])[0] 
#     r_fcst = stats.pearsonr(indices[el],indices[fcst_type])[0]
#     sns.scatterplot(x=el, y='DAM', data=indices, ax=ax, marker='o', linewidth=0.5, 
#                     alpha=0.5, palette=cmap, hue=fcst_type, 
#                     size=fcst_type, sizes=(5,200),
#                     hue_norm=(-10, 50), size_norm=(-10,50))
#     ax.annotate('R_DAM=%.3f, R_IMPV=%.3f'% (r_dam, r_fcst), (-0.05, -0.17))
#     ax.set_xlim([-0.1, 1])
#     ax.set_ylim([-0.2, 0.6])    
# plt.tight_layout()
# plt.show()
# fig.savefig('./Multiplot_Indices_%s.png' % fcst_type)
# 
# # PF
# fcst_type = 'PF'
# cmap = sns.cubehelix_palette(dark=.1, light=.8, rot=-0.4, as_cmap=True)
# fig, axes = plt.subplots(nrows=(indices.columns.shape[0]-3)//3, ncols=3, 
#                          figsize=(12,24), sharey=False)
# sns.set_style('white')
# for (i, el) in enumerate(indices.columns[3:]):
#     ax = axes.flatten('C')[i]
#     r_dam = stats.pearsonr(indices[el],indices['DAM'])[0] 
#     r_fcst = stats.pearsonr(indices[el],indices[fcst_type])[0]
#     sns.scatterplot(x=el, y='DAM', data=indices, ax=ax, marker='o', linewidth=0.5, 
#                     alpha=0.5, palette=cmap, hue=fcst_type, 
#                     size=fcst_type, sizes=(5,200),
#                     hue_norm=(-10, 50), size_norm=(-10,50))
#     ax.annotate('R_DAM=%.3f, R_IMPV=%.3f'% (r_dam, r_fcst), (-0.05, -0.17))
#     ax.set_xlim([-0.1, 1])
#     ax.set_ylim([-0.2, 0.6])    
# plt.tight_layout()
# plt.show()
# fig.savefig('./Multiplot_Indices_%s.png' % fcst_type)
# =============================================================================

# =============================================================================
# #%% Boxplot
# #colors = ["windows blue", "amber", "faded green", "dusty purple"]
# #palette=sns.xkcd_palette(colors)
# #sns.palplot(sns.xkcd_palette(colors))
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8,6), sharey=False)
# for (i, el) in enumerate(indList):
#     ax = axes.flatten('F')[i]
#     sns.set_style('white')
#     sns.boxplot(data=data.loc[indList[i]], ax=ax, width=0.7, palette='deep')
#     ax.set_title(indName[i])
#     ax.set_ylim([-1, 1])
# plt.tight_layout()
# plt.show()
# =============================================================================
    











    
    