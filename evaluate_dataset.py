#!/usr/bin/env python3
"""
Emotion algorithm to evaluate personalization strategies. 


Copyright 2021, J.S. Gómez-Cañón
Licensed under GNU AFFERO GENERAL PUBLIC LICENSE
"""

import argparse
import numpy as np
import pandas as pd
import json
import os
import sys
import pdb
import joblib
from collections import Counter
from scipy.stats import entropy
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
import gc
import datetime
from sklearn.metrics import classification_report, f1_score
import time
from tqdm import tqdm
import subprocess
import shutil
import warnings
warnings.filterwarnings("ignore")


class Evaluator():
    def __init__(self,
                 num_anno,
                 epochs,
                 n_queries,
                 bal_flag):
        """Constructor method
        """
        # dataset_anno = './data/data_24_11_2021.json'
        dataset_anno = './data/data_07_03_2022.json'
        dataset_fn = './data/dataset_feats.csv'
        self.queries = n_queries
        self.dataset = pd.read_csv(dataset_fn, sep=';')
        self.path_to_models = './models/pretrained/'
        self.path_models_users = './models/users_q{}_e{}_bal_anno_entr/'.format(n_queries, epochs)
        self.epochs = epochs
        data = self.load_json(dataset_anno)
        anno = pd.DataFrame(data['annotations'])
        users = pd.DataFrame(data['users'])
        anno['quadrant'] = list(map(self.aro_val_to_quads, anno['arousalValue'].tolist(), anno['valenceValue'].tolist()))

        all_users = [_ for _ in anno.userId.unique().tolist()]
        self.user_anno_dict = {_: anno.externalID[anno.userId == _].tolist() for _ in all_users}
        self.all_users = [k for k,v in self.user_anno_dict.items() if len(v) >= num_anno]
        self.anno = anno[anno.userId.isin(self.all_users)]
        self.all_modes = ['hc', 'mix', 'rand', 'mc']
        self.mod_list = [os.path.join(root, f) for root, dirs, files in os.walk(self.path_to_models) for f in files if f.lower().endswith('.pkl')]
        # self.dict_class = {0: 'Q1', 1: 'Q2', 2: 'Q3', 3: 'Q4'}
        self.seed = np.random.seed(1958)
        self.bal_flag = bal_flag


    def aro_val_to_quads(self, aro, val):
        aro, val = int(aro), int(val)
        if aro == 1 and val == 1:
            quad = 0
        elif aro == 1 and val == -1:
            quad = 1
        elif aro == -1 and val == -1:
            quad = 2
        elif aro == -1 and val == 1:
            quad = 3
        return quad


    def consensus_hc(self, anno, u_id, songs_train):
        # load annotations
        anno.rename(columns={'externalID': 's_id'}, inplace=True)
        anno['s_id'] = anno['s_id'].map(lambda x: x.lower())

        # use only songs from the train batch
        anno = anno[anno.s_id.isin(songs_train)]

        # calculate frequencies for human consensus entropy
        frequencies = {}
        for id_song in anno.s_id.unique().tolist():
            cnt_quad = Counter({0: 0, 1: 0, 2: 0, 3: 0})
            cnt_quad.update(anno[anno.s_id == id_song].quadrant)
            num_anno = anno.loc[anno.s_id == id_song].shape[0]
            cnt_quad = dict(cnt_quad)
            frequencies[id_song] = {k: np.round(v / num_anno, 3) for k, v in cnt_quad.items()}
        
        freqs = pd.DataFrame(frequencies).transpose()
        freqs.index.name = 's_id'
        return freqs


    def create_user(self, u_id, mode):
        # create users folder
        user_path = os.path.join(self.path_models_users, str(u_id), mode +'/')
        try:
            os.makedirs(user_path)
        except FileExistsError:
            if os.path.exists(os.path.join(self.path_models_users, str(u_id), 'f1.csv')):
                print('User has already been evaluated, jumping to next!')
                pass
            else:
                print('User has already created but not evaluated, deleting!')
                subprocess.run(['rm', '-rf', user_path])
                os.makedirs(user_path)
            
        pre_models = self.mod_list
        cp_models = [f.replace(self.path_to_models, user_path) for f in self.mod_list]

        for in_f, out_f in zip(pre_models, cp_models):
            shutil.copy(in_f, out_f)

        return cp_models


    def load_json(self, filename):
        with open(filename, 'r') as f:
            data = f.read()
        data = json.loads(data)
        return data


    def run(self):
        for u_id in tqdm(self.all_users):
            #############################
            # select data for this user
            this_songs = [_.lower() for _ in self.user_anno_dict[u_id]]
            this_dataset = self.dataset[self.dataset.s_id.isin(this_songs)]
            this_anno_user = self.anno[self.anno.userId == u_id][['externalID', 'quadrant']].set_index('externalID')
            this_dict = {k.lower(): v for k, v in this_anno_user.to_dict()['quadrant'].items()}
            this_y = pd.DataFrame.from_dict(this_dict, orient='index').reindex(this_dataset.s_id).rename({0: 'quadrant'})
            # split to train and test data
            gss = GroupShuffleSplit(n_splits=1, train_size=0.85, random_state=self.seed)
            train_idx, test_idx = next(gss.split(this_dataset, this_y, this_y.index))
            X_train, y_train = this_dataset.iloc[train_idx], this_y.iloc[train_idx]
            X_test, y_test = this_dataset.iloc[test_idx], this_y.iloc[test_idx]
            # Normalize features
            X_train_np = StandardScaler().fit_transform(X_train.loc[:, 'F0final_sma_stddev':'mfcc_sma_de[14]_amean'])
            X_train_np = pd.DataFrame(X_train_np, index=X_train.s_id)
            X_test_np = StandardScaler().fit_transform(X_test.loc[:, 'F0final_sma_stddev':'mfcc_sma_de[14]_amean'])
            X_test_np = pd.DataFrame(X_test_np, index=X_test.s_id)

            #calculate hc consensus for this user
            this_hc = self.consensus_hc(self.anno.copy(deep=True), u_id, X_train.s_id.unique().tolist())
            f1_scores = {-1: {'initial': {}}}

            #############################
            # test initial performance before re-trainining
            for i, mod_fn in enumerate(self.mod_list):
                mod = joblib.load(mod_fn)
                # test using testing data
                this_mod_pred = mod.predict(X_test_np.values)
                cl_re = classification_report(y_test.values, this_mod_pred)
                f1_scores[-1]['initial'][mod_fn.split('/')[-1]] = f1_score(y_test.values, this_mod_pred, average='weighted')

            for mode in self.all_modes:
                #############################
                # create each user
                print('Training mode {} for user {}!'.format(mode, u_id))
                this_models = self.create_user(u_id, mode)
                f1_scores[mode] = {}
                this_X_train = X_train.copy(deep=True)
                this_X_train_np = X_train_np.copy(deep=True)
                this_y_train = y_train.copy(deep=True)
                this_hc_mode = this_hc.copy(deep=True)

                # for e in range(self.epochs):
                for e in tqdm(range(self.epochs)):
                    f1_scores[mode][e] = {}
                    #############################
                    # choose instances according to each consensus entropy approach
                    if mode == 'hc':
                        # human consensus (HC)
                        ent_hc = entropy(this_hc_mode, axis=1)
                        q_ind = np.argsort(ent_hc)[::-1]
                        q_songs_full = this_hc_mode.iloc[q_ind].index.tolist()
                        if self.bal_flag:
                            # try balancing
                            q_list_bal = []
                            list_bal = this_anno_user.quadrant.unique().tolist() * int(self.queries / len(this_anno_user.quadrant.unique().tolist()))
                            for song in q_songs_full:
                                this_class = this_dict[song]
                                if this_class in list_bal:
                                    q_list_bal.append(song)
                                    list_bal.remove(this_class)
                            q_songs = q_list_bal
                            # if a class is no longer represented fill with other samples of high entropy
                            if len(q_songs) != self.queries:
                                s_to_fill = self.queries - len(q_songs)
                                trim = [s for s in q_songs_full if s not in q_songs]
                                q_songs.extend(trim[:s_to_fill])

                        else:
                            q_songs = q_songs_full[:self.queries]
                            # remove songs from this batch
                        this_hc_mode = this_hc_mode[~this_hc_mode.index.isin(q_songs)]

                    elif mode == 'mc':
                        # machine consensus (MC)
                        pred_prob = []
                        for i, mod_fn in enumerate(this_models):
                            mod = joblib.load(mod_fn)
                            y_probs = mod.predict_proba(this_X_train_np)
                            # summarize with mean across all samples
                            y_probs = pd.DataFrame(y_probs, index=this_X_train.s_id).groupby(['s_id']).mean()
                            pred_prob.append(y_probs)
                            gc.collect()
                        consensus_prob = pd.DataFrame(np.mean(np.array(pred_prob), axis=0), 
                                                      columns=[0, 1, 2, 3],
                                                      index=y_probs.index)

                        # entropy calculation
                        ent = entropy(consensus_prob, axis=1)
                        # select songs with max entropy for self.queries amount
                        q_ind = np.argsort(ent)[::-1]
                        # select songs from the average of output probabilities
                        q_songs_full = y_probs.iloc[q_ind].index.tolist()
                        if self.bal_flag:
                            # try balancing
                            q_list_bal = []
                            list_bal = this_anno_user.quadrant.unique().tolist() * int(self.queries / len(this_anno_user.quadrant.unique().tolist()))
                            for song in q_songs_full:
                                this_class = this_dict[song]
                                if this_class in list_bal:
                                    q_list_bal.append(song)
                                    list_bal.remove(this_class)
                            q_songs = q_list_bal
                            # if a class is no longer represented fill with other samples of high entropy
                            if len(q_songs) != self.queries:
                                s_to_fill = self.queries - len(q_songs)
                                trim = [s for s in q_songs_full if s not in q_songs]
                                q_songs.extend(trim[:s_to_fill])
                        else:
                            q_songs = q_songs_full[:self.queries]

                    elif mode == 'mix':
                        # hybrid consensus (HC)
                        pred_prob = []
                        for i, mod_fn in enumerate(this_models):
                            mod = joblib.load(mod_fn)
                            y_probs = mod.predict_proba(this_X_train_np)
                            # summarize with mean across all samples
                            y_probs = pd.DataFrame(y_probs, index=this_X_train.s_id).groupby(['s_id']).mean()
                            pred_prob.append(y_probs)
                            gc.collect()
                        consensus_prob_mc = pd.DataFrame(np.mean(np.array(pred_prob), axis=0), 
                                                         columns=[0, 1, 2, 3],
                                                         index=y_probs.index)

                        # include human consensus
                        mix_consensus = pd.concat([consensus_prob_mc, this_hc_mode])
                        mix_consensus = mix_consensus.groupby('s_id').mean()

                        # entropy calculation
                        ent_mix = entropy(mix_consensus, axis=1)
                        q_ind = np.argsort(ent_mix)[::-1]
                        q_songs_full = mix_consensus.iloc[q_ind].index.tolist()
                        if self.bal_flag:
                            # try balancing
                            q_list_bal = []
                            list_bal = this_anno_user.quadrant.unique().tolist() * int(self.queries / len(this_anno_user.quadrant.unique().tolist()))
                            for song in q_songs_full:
                                this_class = this_dict[song]
                                if this_class in list_bal:
                                    q_list_bal.append(song)
                                    list_bal.remove(this_class)
                            q_songs = q_list_bal
                            # if a class is no longer represented fill with other samples of high entropy
                            if len(q_songs) != self.queries:
                                s_to_fill = self.queries - len(q_songs)
                                trim = [s for s in q_songs_full if s not in q_songs]
                                q_songs.extend(trim[:s_to_fill])

                        else:
                            q_songs = q_songs_full[:self.queries]
                        # remove songs from this batch
                        this_hc_mode = this_hc_mode[~this_hc_mode.index.isin(q_songs)]

                    elif mode == 'rand':
                        # random baseline (RAND)
                        pos_songs = this_X_train.s_id.unique().tolist()
                        np.random.shuffle(pos_songs)
                        q_songs = pos_songs[:self.queries]

                    #############################
                    # retrain models with this batch
                    X_batch = this_X_train_np[this_X_train_np.index.isin(q_songs)]
                    y_batch = this_y_train[this_y_train.index.isin(q_songs)]

                    # assert length of choice
                    if (len(X_batch.index.unique().tolist()) != self.queries) or (len(y_batch.index.unique().tolist()) != self.queries):
                        # debugging
                        pdb.set_trace()
                        # X_batch = this_X_train_np[this_X_train_np.index.isin(q_songs)]
                        # y_batch = this_y_train[this_y_train.index.isin(q_songs)]

                    for i, mod_fn in enumerate(this_models):
                        mod = joblib.load(mod_fn)
                        try:
                            if mod_fn.find('_xgb') > 0:
                                mod.fit(X_batch.values, y_batch.values.ravel(), eval_metric='auc', xgb_model=mod.get_booster()) 
                            else:
                                mod.partial_fit(X_batch.values, y_batch.values.ravel())
                        except:
                            pdb.set_trace()
                        joblib.dump(mod, mod_fn)

                    #############################
                    # evaluate models and save f-scores
                    for i, mod_fn in enumerate(this_models):
                        mod = joblib.load(mod_fn)
                        # test using testing data
                        this_mod_pred = mod.predict(X_test_np.values)
                        cl_re = classification_report(y_test.values, this_mod_pred)
                        f1_scores[mode][e][mod_fn.split('/')[-1]] = f1_score(y_test.values, this_mod_pred, average='weighted')

                    #############################
                    # remove batch from training instances
                    this_X_train = this_X_train[~this_X_train.s_id.isin(q_songs)]
                    this_X_train_np = this_X_train_np[~this_X_train_np.index.isin(q_songs)]
                    this_y_train = this_y_train[~this_y_train.index.isin(q_songs)]
                    # print(this_X_train.shape)

                # clean memory
                del this_X_train
                del this_X_train_np
                del this_y_train
                del this_hc_mode
                gc.collect()

            #############################
            # save evaluation history
            r_f = {(o_k, i_k): v for o_k, i_d in f1_scores.items() for i_k, v in i_d.items()}
            f1_df = pd.DataFrame(r_f).transpose()
            f1_fn = os.path.join(self.path_models_users, str(u_id), 'f1.csv')
            f1_df.to_csv(f1_fn)
            

if __name__ == "__main__":
    # usage: python3 evaluate_dataset.py -n NUM_ANNOTATIONS -e NUM_EPOCHS -q NUM_QUERIES
    # example: python3 evaluate_dataset.py -n 80 -e 10 -q 5
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-q',
                        '--queries',
                        help='Select number of queries per iteration (int)',
                        action='store',
                        required=True,
                        type=int,
                        dest='queries')
    parser.add_argument('-e',
                        '--epochs',
                        help='Select number of epochs to perform (int)',
                        action='store',
                        required=True,
                        type=int,
                        dest='epochs')
    parser.add_argument('-b',
                        '--balanced',
                        help='Attempt to balance classes [True, False]',
                        action='store',
                        required=True,
                        type=bool,
                        dest='balanced')
    parser.add_argument('-n',
                        '--num_anno',
                        help='Select minimum number of annotations per user (int)',
                        action='store',
                        required=True,
                        type=int,
                        dest='num_anno')
    args = parser.parse_args()

    if args.queries * args.epochs >= 0.85 * args.num_anno:
        print('The number of queries or epochs results in more queries than annotated songs!')
        sys.exit()


    if args.balanced:
        if args.queries % 4 != 0:
            print('If you want to balance the entropy calculator, select number of queries as multiple of classes (i.e., 4 or 8)')

    evaluator = Evaluator(args.num_anno, args.epochs, args.queries, args.balanced)

    evaluator.run()


    print('Process lasted {} minutes!'.format((time.time()-start) / 60))


