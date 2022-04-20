#!/usr/bin/env python3
"""
Plot results from the evaluation of the dataset.


Copyright 2021, J.S. Gómez-Cañón
Licensed under GNU AFFERO GENERAL PUBLIC LICENSE
"""
import numpy as np
import pandas as pd
import json
import os
import sys
import pdb
import matplotlib.pyplot as plt
import itertools
from scipy import stats
import statsmodels.api as sm

from collections import Counter
import krippendorff

def make_plots(df, user_list, cons_types, mod_types, path):
    # confidence intervals
    c_ints = {90: 1.64, 95: 1.96, 99: 2.33, 99.5: 2.58}
    c_lev = c_ints[95]

    new_data_cols = {'initial': 'init'}
    df = df.rename(columns=new_data_cols)

    # pdb.set_trace()
    plt.figure(figsize=(12, 4.2))
    plt.rc('xtick', labelsize=9)
    plt.rc('ytick', labelsize=9)
    for i, mod in enumerate(mod_types):
        plt.subplot(1, 3, i + 1)
        for cons in cons_types:
            mean = df[(df.model == mod) & (df.consensus == cons)].loc[:, 'init':].mean()
            std = df[(df.model == mod) & (df.consensus == cons)].loc[:, 'init':].std()
            cnt = df[(df.model == mod) & (df.consensus == cons)].loc[:, 'init':].count()

            c_i = c_lev * std / np.sqrt(cnt)
            # texts
            mod_sht = mod.split('_')[-1]
            lbl = '{}_{}'.format(mod_sht, cons)
            plt.plot(mean.transpose(), label=cons)

            plt.fill_between(mean.index, mean.transpose() + c_i, mean.transpose() - c_i, alpha=0.2)
            plt.title('Model: {}'.format(mod_sht.upper()))
            plt.ylabel('F1 score')
            plt.xlabel('Epochs')
            plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'comparison_models.svg'))
    plt.show()

def get_reformed_dict(in_dict):
    reformed_dict = {} 
    for outerKey, innerDict in in_dict.items(): 
        for innerKeyA, inDict in innerDict.items(): 
            for innerKeyB, values in inDict.items():
                reformed_dict[(outerKey, innerKeyA, innerKeyB)] = values 
    return reformed_dict


def get_stats(df, user_list, cons_types, mod_types, path):
    data_cols = df.columns[5:].tolist()
    cons_comb = list(itertools.combinations(cons_types, 2))

    print('one-sided t-test by model by point')
    t_test = {}
    for mod in mod_types:
        t_test[mod] = {}
        # print('Model {}'.format(mod))
        for cons_a, cons_b in cons_comb:
            txt_cons = '{} vs {}'.format(cons_a, cons_b)
            t_test[mod][txt_cons] = {}
            for it in data_cols:
                data_a = df[(df.model == mod) & (df.consensus == cons_a)]
                data_b = df[(df.model == mod) & (df.consensus == cons_b)]
                tval, pval = stats.ttest_ind(data_a.loc[:, it], data_b.loc[:, it], alternative='greater')
                t_test[mod][txt_cons][it] = {}
                
                stat_sig = False
                if pval < 0.05:
                    stat_sig = True
                t_test[mod][txt_cons][it].update({'t_val': tval, 'p_val': pval, 'n': data_a.shape[0], 'stat_sig': stat_sig})

    t_test_df = pd.DataFrame(get_reformed_dict(t_test)).transpose()
    t_test_df.index.set_names(['model', 'comparison', 'iteration'], inplace=True)
    t_test_df.reset_index(inplace=True)

    print(t_test_df[(t_test_df.stat_sig == True) & (t_test_df.iteration == data_cols[-1])])
    t_test_df.to_csv(os.path.join(path, 't_test.csv'))

    print('per user calculation of lineal tendency')
    slopes = {}
    for u_id in user_list:
        slopes[u_id] = {}
        for mod in mod_types:
            slopes[u_id][mod] = {}
            for cons in cons_types:
                data = df[(df.model == mod) & (df.user == u_id) & (df.consensus == cons)]
                x_data = np.repeat(np.array([data_cols]).astype(int), data.shape[0], axis=0)
                y_data = data.loc[:, data_cols].values
                z = np.polyfit(x_data.flatten(), y_data.flatten(), 1)
                p = np.poly1d(z)
                slope = z[0]
                pos_slope = False
                if slope >= 0:
                    pos_slope = True
                slopes[u_id][mod][cons] = {}
                slopes[u_id][mod][cons].update({'slope': slope, 'pos': pos_slope})

    slope_df = pd.DataFrame(get_reformed_dict(slopes)).transpose()
    slope_df.index.set_names(['user', 'model', 'consensus'], inplace=True)
    slope_df.reset_index(inplace=True)

    comp_pre = df.groupby(by=['user', 'model', 'consensus']).mean()
    comp_pre.reset_index(inplace=True)
    comp_pre['pos'] = comp_pre[data_cols[0]] > comp_pre[data_cols[-1]]

    cnt_slopes = {}
    pre_cnt = {}
    for mod in mod_types:
        cnt_slopes[mod] = {}
        pre_cnt[mod] = {}
        for cons in cons_types:
            cnt_slopes[mod][cons] = slope_df[(slope_df.model == mod) & (slope_df.consensus == cons)].pos.sum()
            pre_cnt[mod][cons] = comp_pre[(comp_pre.model == mod) & (comp_pre.consensus == cons)].pos.sum()

    hist_slopes = pd.DataFrame.from_dict(cnt_slopes).transpose()
    hist_slopes = hist_slopes[['hc', 'mc', 'mix', 'rand']]

    hist_pre = pd.DataFrame.from_dict(pre_cnt).transpose()
    hist_pre = hist_pre[['hc', 'mc', 'mix', 'rand']]

    norm_hist_slopes = hist_slopes / len(user_list)
    norm_hist_slopes = norm_hist_slopes[['hc', 'mc', 'mix', 'rand']]

    hist_slopes.plot(kind='bar', rot=0, alpha=0.7, figsize=(6, 2.5))
    plt.xlabel('Model')
    plt.ylabel('# of users')
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'effective.svg'))
    plt.show()

    return slope_df

def analyze_users(slope_df, anno, users, scores_df):
    # analyze user behavior
    slope_df['pos'] += 0
    sum_us = slope_df.groupby(['user'])['pos'].sum()
    lo_users = sum_us[sum_us < sum_us.mean() - sum_us.std()].index.tolist()
    med_users = sum_us[(sum_us > sum_us.mean() - sum_us.std()) & (sum_us < sum_us.mean() + sum_us.std())].index.tolist()
    hi_users = sum_us[sum_us > sum_us.mean() + sum_us.std()].index.tolist()
    print('Amount of effective personalization ensembles per user:')
    print(sum_us)
    # encode data
    tags = ['joy', 'power', 'surprise', 'anger', 'tension', 'fear', 'sadness', 'bitterness', 'peace', 'tenderness', 'transcendence']
    tags_enc = {v: k for k, v in enumerate(tags)}
    anno['quadrant'] = list(map(aro_val_to_quads, anno['arousalValue'].tolist(), anno['valenceValue'].tolist()))
    anno['moodValueEnc'] = anno['moodValue'].map(tags_enc)
    anno['arousalValue'] = anno['arousalValue'].astype(int)
    anno['valenceValue'] = anno['valenceValue'].astype(int)
    # split annotations
    anno_hi = anno[anno.userId.isin(hi_users)]
    anno_lo = anno[anno.userId.isin(lo_users)]
    anno_med = anno[anno.userId.isin(med_users)]

    # calculate agreement
    alpha_quad, alpha_aro, alpha_val, alpha_emo = get_info_per_song(anno_hi)
    print('Agreement for high personalization:\nQuadrant: {}\nArousal: {}\nValence: {}\nEmotion: {}\n'.format(alpha_quad, alpha_aro, alpha_val, alpha_emo))
    alpha_quad, alpha_aro, alpha_val, alpha_emo = get_info_per_song(anno_med)
    print('Agreement for medium personalization:\nQuadrant: {}\nArousal: {}\nValence: {}\nEmotion: {}\n'.format(alpha_quad, alpha_aro, alpha_val, alpha_emo))
    alpha_quad, alpha_aro, alpha_val, alpha_emo = get_info_per_song(anno_lo)
    print('Agreement for low personalization:\nQuadrant: {}\nArousal: {}\nValence: {}\nEmotion: {}\n'.format(alpha_quad, alpha_aro, alpha_val, alpha_emo))
    pdb.set_trace()
    return hi_users, med_users, lo_users


def anova_tests(df, hi_users, med_users, lo_users):
    # organize data for analysis
    col_name = {str(_): 'epoch_{}'.format(_) for _ in range(15)}
    df.rename(columns=col_name, inplace=True)

    df['type'] = ''
    df.loc[df.user.isin(lo_users), 'type'] = 'low'
    df.loc[df.user.isin(med_users), 'type'] = 'medium'
    df.loc[df.user.isin(hi_users), 'type'] = 'high'
    pdb.set_trace()

    # # two way anova with repeated measures
    # print('-----\nAnova and Tukey HSD for complete data\n')
    # mod = sm.stats.AnovaRM(df, 'epoch_14', 'user', within=['model', 'consensus', 'iteration'])
    # res = mod.fit()
    # print('ANOVA')
    # print(res)
    # print('Tukey HSD')
    # print(sm.stats.multicomp.pairwise_tukeyhsd(df.epoch_14.tolist(), df.model.tolist()))
    # print(sm.stats.multicomp.pairwise_tukeyhsd(df.epoch_14.tolist(), df.consensus.tolist()))
    # print(sm.stats.multicomp.pairwise_tukeyhsd(df.epoch_14.tolist(), df.type.tolist()))
    # print(sm.stats.multicomp.pairwise_tukeyhsd(df.epoch_14.tolist(), df.iteration.tolist()))

    print('Separation by model type: SGD, GNB, XGB')
    df_gnb = df[df.model == 'classifier_gnb']
    df_sgd = df[df.model == 'classifier_sgd']
    df_xgb = df[df.model == 'classifier_xgb']
    # Model GNB
    mod = sm.stats.AnovaRM(df_gnb, 'epoch_14', 'user', within=['consensus', 'iteration'])
    res = mod.fit()
    print('ANOVA GNB')
    print(res)
    print('Tukey HSD GNB')
    print(sm.stats.multicomp.pairwise_tukeyhsd(df_gnb.epoch_14.tolist(), df_gnb.consensus.tolist()))
    print(sm.stats.multicomp.pairwise_tukeyhsd(df_gnb.epoch_14.tolist(), df_gnb.type.tolist()))
    print(sm.stats.multicomp.pairwise_tukeyhsd(df_gnb.epoch_14.tolist(), df_gnb.iteration.tolist()))
    # Model SGD
    mod = sm.stats.AnovaRM(df_sgd, 'epoch_14', 'user', within=['consensus', 'iteration'])
    res = mod.fit()
    print('ANOVA SGD')
    print(res)
    print('Tukey HSD SGD')
    print(sm.stats.multicomp.pairwise_tukeyhsd(df_sgd.epoch_14.tolist(), df_sgd.consensus.tolist()))
    print(sm.stats.multicomp.pairwise_tukeyhsd(df_sgd.epoch_14.tolist(), df_sgd.type.tolist()))
    print(sm.stats.multicomp.pairwise_tukeyhsd(df_sgd.epoch_14.tolist(), df_sgd.iteration.tolist()))
    # Model XGB
    mod = sm.stats.AnovaRM(df_xgb, 'epoch_14', 'user', within=['consensus', 'iteration'])
    res = mod.fit()
    print('ANOVA XGB')
    print(res)
    print('Tukey HSD XGB')
    print(sm.stats.multicomp.pairwise_tukeyhsd(df_xgb.epoch_14.tolist(), df_xgb.consensus.tolist()))
    print(sm.stats.multicomp.pairwise_tukeyhsd(df_xgb.epoch_14.tolist(), df_xgb.type.tolist()))
    print(sm.stats.multicomp.pairwise_tukeyhsd(df_xgb.epoch_14.tolist(), df_xgb.iteration.tolist()))


    # marginal mean plots
    plt.title('Marginal mean plot for model:consensus')
    plt.plot(df_gnb.groupby(['consensus']).mean().epoch_14, label='GNB')
    plt.plot(df_sgd.groupby(['consensus']).mean().epoch_14, label='SGD')
    plt.plot(df_xgb.groupby(['consensus']).mean().epoch_14, label='XGB')
    plt.legend()
    plt.show()

    plt.title('Marginal mean plot for model:iteration')
    plt.plot(df_gnb.groupby(['iteration']).mean().epoch_14, label='GNB')
    plt.plot(df_sgd.groupby(['iteration']).mean().epoch_14, label='SGD')
    plt.plot(df_xgb.groupby(['iteration']).mean().epoch_14, label='XGB')
    plt.legend()
    plt.show()

    df_gnb.boxplot('epoch_14', by=['consensus', 'iteration'])
    plt.show()
    pdb.set_trace()



def aro_val_to_quads(aro, val):
    aro, val = int(aro), int(val)
    if aro == 1 and val == 1:
        quad = 1
    elif aro == 1 and val == -1:
        quad = 2
    elif aro == -1 and val == -1:
        quad = 3
    elif aro == -1 and val == 1:
        quad = 4
    return quad

def load_json(filename):
    with open(filename, 'r') as f:
        data = f.read()
    data = json.loads(data)
    return data

def get_info_per_song(anno):
    quadrant = pd.pivot_table(anno,
                             index=['userId'],
                             columns=['externalID'],
                             values=['quadrant'])
    alpha_quad = krippendorff.alpha(reliability_data=quadrant, level_of_measurement='nominal')
    arousal = pd.pivot_table(anno,
                             index=['userId'],
                             columns=['externalID'],
                             values=['arousalValue'])
    alpha_aro = krippendorff.alpha(reliability_data=arousal, level_of_measurement='nominal')
    valence = pd.pivot_table(anno,
                             index=['userId'],
                             columns=['externalID'],
                             values=['valenceValue'])
    alpha_val = krippendorff.alpha(reliability_data=valence, level_of_measurement='nominal')
    emotion = pd.pivot_table(anno,
                             index=['userId'],
                             columns=['externalID'],
                             values=['moodValueEnc'])
    alpha_emo = krippendorff.alpha(reliability_data=emotion, level_of_measurement='nominal')
    return alpha_quad, alpha_aro, alpha_val, alpha_emo


if __name__ == "__main__":
    # usage: python3 analysis.py
    path_models_users = './models/users/users_q4_e15_bal_entr/'


    # load data and format for plotting
    res_list = [os.path.join(root, f) for root, dirs, files in os.walk(path_models_users) for f in files if f.lower().endswith('f1.csv')]
    df_list = [pd.read_csv(_, index_col=[0, 1]) for _ in res_list]
    # user_list = [_.split('/')[3] for _ in res_list]
    user_list = [_.split('/')[-2] for _ in res_list]
    

    modes = ['hc', 'mix', 'mc', 'rand']
    models = ['classifier_gnb', 'classifier_sgd', 'classifier_xgb']

    struc_df = []
    for u_id, df in zip(user_list, df_list):
        list_in_modes = []
        for mo in modes:
            this_mo = pd.concat([df.loc['-1'].transpose(), df.loc[mo].transpose()], axis=1).reset_index()
            this_mo[['model', 'iteration', 'format']] = this_mo['index'].str.split('.', expand=True)
            this_mo.drop(columns=['index', 'format'], inplace=True)
            this_mo['consensus'] = mo
            this_mo['user'] = u_id
            this_mo.set_index(['user', 'model', 'consensus', 'iteration'], inplace=True)
            list_in_modes.append(this_mo)
        this_df = pd.concat(list_in_modes)
        struc_df.append(this_df)

    struc_df = pd.concat(struc_df)

    make_plots(struc_df.reset_index(), user_list, modes, models, path_models_users)

    sl_df = get_stats(struc_df.reset_index(), user_list, modes, models, path_models_users)

    # dataset_anno = './data/data_24_11_2021.json'
    dataset_anno = './data/data_07_03_2022.json'
    data = load_json(dataset_anno)
    anno = pd.DataFrame(data['annotations'])
    users = pd.DataFrame(data['users'])

    hi_users, med_users, lo_users = analyze_users(sl_df, anno, users, struc_df.reset_index())


    anova_tests(struc_df.reset_index(), hi_users, med_users, lo_users)
