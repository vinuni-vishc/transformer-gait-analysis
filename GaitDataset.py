"""
The dataset and this implementation is adapted from Stanford's mobile-gaitlab: https://github.com/stanfordnmbl/mobile-gaitlab
"""
import pandas as pd
import numpy as np
import pickle
import torch
import collections
from config import *

class GaitDataset:

    def __init__(self, target_metric, doublesided = False, hef = False):
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"
        self.device = torch.device(dev)

        self.doublesided = doublesided
        self.hef = hef
        self.target_metric = target_metric
        self.ids_nonmissing_target = set()
        self.processed_data = self.load_processed_data()
        self.video_segments = self.load_segments()
        self.datasplit_ids = self.load_datasplit_ids()
        self.x_columns = self.load_x_columns()
        self.target_dict = self.load_target_dict()
        self.target_min = None
        self.target_range = None
        self.all_dataset = self.load_all_data()
        self.train_dataset = self.load_train_data()
        self.validation_dataset = self.load_validation_data()
        self.test_dataset = self.load_test_data()
        self.train_weights, self.validation_weights, self.c_i_factor = self.load_weights()

    def load_processed_data(self):
        alldata_processed = pd.read_csv(
            DATA_PATH + '/alldata_processed_with_dev_residual.csv')
        alldata_processed['videoid'] = alldata_processed['videoid'].apply(
            lambda x: int(x))

        if self.doublesided:
            alldata_processed['target_count'] = alldata_processed.groupby('videoid')[self.target_metric].transform(lambda x: x.count())
            alldata_processed = alldata_processed[alldata_processed[self.target_metric].notnull()]
            alldata_processed = alldata_processed[alldata_processed['target_count'] == 2]
        else:
            alldata_processed = alldata_processed[alldata_processed[self.target_metric].notnull(
            )]
            alldata_processed = alldata_processed.groupby(
                ['videoid', 'side']).head(1)
        self.ids_nonmissing_target = set(alldata_processed['videoid'].unique())

        return alldata_processed

    def load_segments(self):
        if self.doublesided:
            with open(DATA_PATH + '/all_processed_video_segments_doublesided.pickle', 'rb') as handle:
                processed_video_segments = pickle.load(handle) 
        else:
            with open(DATA_PATH + '/all_processed_video_segments.pickle', 'rb') as handle:
                processed_video_segments = pickle.load(handle)
        return processed_video_segments

    def load_datasplit_ids(self):
        datasplit_df = pd.read_csv(
            DATA_PATH + '/train_test_valid_id_split.csv')
        datasplit_df['videoid'] = datasplit_df['videoid'].apply(
            lambda x: int(x))
        all_ids = set(datasplit_df['videoid']).intersection(
            self.ids_nonmissing_target)
        train_ids = set(datasplit_df[datasplit_df['dataset'] == 'train']['videoid']).intersection(
            self.ids_nonmissing_target)
        validation_ids = set(datasplit_df[datasplit_df['dataset'] == 'validation']['videoid']).intersection(
            self.ids_nonmissing_target)
        test_ids = set(datasplit_df[datasplit_df['dataset'] == 'test']['videoid']).intersection(
            self.ids_nonmissing_target)

        return {
            'all': all_ids,
            'train': train_ids,
            'validation': validation_ids,
            'test': test_ids
        }

    def load_x_columns(self):
        if self.doublesided:
            if self.hef:
                return [[2*LANK, 2*LANK+1, 2*LKNE, 2*LKNE+1,
                        2*LHIP, 2*LHIP+1, 2*LBTO, 2*LBTO+1, 50, 52, 54, 56],
                        [2*RANK, 2*RANK+1, 2*RKNE, 2*RKNE+1,
                        2*RHIP, 2*RHIP+1, 2*RBTO, 2*RBTO+1, 51, 53, 55, 57]]
            else:
                return [[2*LANK, 2*LANK+1, 2*LKNE, 2*LKNE+1,
                        2*LHIP, 2*LHIP+1, 2*LBTO, 2*LBTO+1],
                        [2*RANK, 2*RANK+1, 2*RKNE, 2*RKNE+1,
                        2*RHIP, 2*RHIP+1, 2*RBTO, 2*RBTO+1]]
        else:
            if self.hef:
                return [2*LANK, 2*LANK+1, 2*LKNE, 2*LKNE+1,
                    2*LHIP, 2*LHIP+1, 2*LBTO, 2*LBTO+1,
                    2*RANK, 2*RANK+1, 2*RKNE, 2*RKNE+1,
                    2*RHIP, 2*RHIP+1, 2*RBTO, 2*RBTO+1,
                    50, 51, 52, 53, 54, 55, 56, 57]
            else:
                return [2*LANK, 2*LANK+1, 2*LKNE, 2*LKNE+1,
                        2*LHIP, 2*LHIP+1, 2*LBTO, 2*LBTO+1,
                        2*RANK, 2*RANK+1, 2*RKNE, 2*RKNE+1,
                        2*RHIP, 2*RHIP+1, 2*RBTO, 2*RBTO+1]

    def load_target_dict(self):
        if self.doublesided:
            target_dict_L = {}
            target_dict_R = {}
            for i in range(len(self.processed_data)):
                row = self.processed_data.iloc[i]
                if row['side'] == 'L':
                    target_dict_L[row['videoid']] = row[self.target_metric]
                if row['side'] == 'R':
                    target_dict_R[row['videoid']] = row[self.target_metric]
            return {"L": target_dict_L, "R": target_dict_R}
        else:
            target_dict = {}
            for i in range(len(self.processed_data)):
                row = self.processed_data.iloc[i]
                target_dict[row['videoid']] = row[self.target_metric]
            return target_dict

    def load_all_data(self):
        X = [t[2] for t in self.video_segments if t[0]
             in self.datasplit_ids['all']]
        y = []

        if self.doublesided:
            X = np.stack(X)
            X = np.concatenate([X[:,:,self.x_columns[0]],X[:,:,self.x_columns[1]]])
            if not self.hef:
                X = np.reshape(X, (-1, NUM_FRAME, 4, 2))
            y_L = [self.target_dict['L'][t[0]]
                     for t in self.video_segments if t[0] in self.datasplit_ids['all']]
            y_R = [self.target_dict['R'][t[0]]
                     for t in self.video_segments if t[0] in self.datasplit_ids['all']]
            y = np.array (y_L + y_R)
        else:
            X = np.stack(X)[:, :, self.x_columns]
            if not self.hef:
                X = np.reshape(X, (-1, NUM_FRAME, 8, 2))
            y = np.array([self.target_dict[t[0]]
                     for t in self.video_segments if t[0] in self.datasplit_ids['all']]) 
        
        X = torch.tensor(X, dtype = torch.float32, device = self.device)

        return (X, y)

    def load_train_data(self):
        X_train = [t[2] for t in self.video_segments if t[0]
                   in self.datasplit_ids['train']]
        y_train = []

        if self.doublesided:
            X_train = np.stack(X_train)
            X_train = np.concatenate([X_train[:,:,self.x_columns[0]],X_train[:,:,self.x_columns[1]]])
            if not self.hef:
                X_train = np.reshape(X_train, (-1, NUM_FRAME, 4, 2))
            y_train_L = [self.target_dict['L'][t[0]]
                     for t in self.video_segments if t[0] in self.datasplit_ids['train']]
            y_train_R = [self.target_dict['R'][t[0]]
                     for t in self.video_segments if t[0] in self.datasplit_ids['train']]
            y_train = np.array (y_train_L + y_train_R)
        else:
            X_train = np.stack(X_train)[:, :, self.x_columns]
            if not self.hef:
                X_train = np.reshape(X_train, (-1, NUM_FRAME, 8, 2))
            y_train = np.array([self.target_dict[t[0]]
                     for t in self.video_segments if t[0] in self.datasplit_ids['train']])
            
        X_train = torch.tensor(X_train, dtype = torch.float32, device=self.device)

        self.target_min = np.min(y_train, axis=0)
        self.target_range = np.max(y_train, axis=0) - np.min(y_train, axis=0)
        y_train = (y_train - self.target_min)/self.target_range

        return (X_train, y_train)

    def load_validation_data(self):
        X_validation = [t[2] for t in self.video_segments if t[0]
                        in self.datasplit_ids['validation']]
        y_validation = []

        if self.doublesided:
            X_validation = np.stack(X_validation)
            X_validation = np.concatenate([X_validation[:,:,self.x_columns[0]], X_validation[:,:,self.x_columns[1]]])
            if not self.hef:
                X_validation = np.reshape(X_validation, (-1, NUM_FRAME, 4, 2))
            y_validation_L = [self.target_dict['L'][t[0]]
                     for t in self.video_segments if t[0] in self.datasplit_ids['validation']]
            y_validation_R = [self.target_dict['R'][t[0]]
                     for t in self.video_segments if t[0] in self.datasplit_ids['validation']]
            y_validation = np.array (y_validation_L + y_validation_R)
        else:
            X_validation = np.stack(X_validation)[:, :, self.x_columns]
            if not self.hef:
                X_validation = np.reshape(X_validation, (-1, NUM_FRAME, 8, 2))
            y_validation = np.array([self.target_dict[t[0]]
                                for t in self.video_segments if t[0] in self.datasplit_ids['validation']])
            
        X_validation = torch.tensor(X_validation, dtype = torch.float32, device=self.device)
        
        y_validation = (y_validation - self.target_min)/self.target_range

        return (X_validation, y_validation)

    def load_test_data(self):
        X_test = [t[2] for t in self.video_segments if t[0]
                  in self.datasplit_ids['test']]
        y_test = []

        if self.doublesided:
            X_test = np.stack(X_test)
            X_test = np.concatenate([X_test[:,:,self.x_columns[0]], X_test[:,:,self.x_columns[1]]])
            if not self.hef:
                X_test = np.reshape(X_test, (-1, NUM_FRAME, 4, 2))
            y_test_L = [self.target_dict['L'][t[0]]
                     for t in self.video_segments if t[0] in self.datasplit_ids['test']]
            y_test_R = [self.target_dict['R'][t[0]]
                     for t in self.video_segments if t[0] in self.datasplit_ids['test']]
            y_test = np.array (y_test_L + y_test_R)
        else:
            X_test = np.stack(X_test)[:, :, self.x_columns]
            if not self.hef:
                X_test = np.reshape(X_test, (-1, NUM_FRAME, 8, 2))
            y_test = np.array([self.target_dict[t[0]]
                                for t in self.video_segments if t[0] in self.datasplit_ids['test']])
            
        X_test = torch.tensor(X_test, dtype = torch.float32, device=self.device)

        return (X_test, y_test)
    
    def load_weights(self):
        count_dict = collections.Counter(np.array([t[0] for t in self.video_segments]))

        train_weights = [1./count_dict[t[0]] for t in self.video_segments if t[0] in self.datasplit_ids['train']]
        train_weights = train_weights + train_weights
        train_weights = np.array(train_weights).reshape(-1,1)

        validation_weights = [1./count_dict[t[0]] for t in self.video_segments if t[0] in self.datasplit_ids['validation']]
        validation_weights = validation_weights + validation_weights
        validation_weights = np.array(validation_weights).reshape(-1,1)

        c_i_factor = np.mean(np.vstack([train_weights, validation_weights])) 

        train_weights = torch.tensor(train_weights, dtype = torch.float32, device=self.device)
        validation_weights = torch.tensor(validation_weights, dtype = torch.float32, device=self.device)

        return train_weights, validation_weights, c_i_factor