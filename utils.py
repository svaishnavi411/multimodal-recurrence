'''
Contains the utility functions
'''

import numpy as np
import config as config

import sys
sys.path.append('../')

from sksurv.linear_model import CoxnetSurvivalAnalysis
from loader import Dataset

available_models = ['genomics', 'pyradiomics',
                    'densenet', 'intermediate_gp',
                    'intermediate_gd', 'late_gp',
                    'late_gd']

def run_coxnet(l1_ratio, n_alphas, x_train, y_train, x_test, y_test):

    coxnet = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, n_alphas=n_alphas)
    coxnet.fit(x_train, y_train)
    outputs = coxnet.predict(x_test)
    score = coxnet.score(x_test, y_test)
    return outputs, score

def get_data(split=0, location=config.csv_location, mode='valid'):
    '''
    use mode = 'test' for testing
    '''

    print('Loading data for mode ' + mode + ' from location ' + location)
    X_train, y_train, y_train2 = [], [], []
    with open(location + 'train_' + str(split) + '.csv', 'r') as curr_file:
        for row in curr_file:
            a, b, c = row.split('\t')
            X_train.append(a.strip())
            y_train.append(int(b.strip()))
            y_train2.append(int(c.strip()))

    X_test, y_test, y_test2 = [], [], []
    with open(location + mode + '_' + str(split) + '.csv', 'r') as curr_file:
        for row in curr_file:
            a, b, c = row.split('\t')
            X_test.append(a.strip())
            y_test.append(int(b.strip()))
            y_test2.append(int(c.strip()))
 
    return X_train, X_test, y_train, y_test, y_train2, y_test2

def get_structured_array(data_bool, data_value):
    all_bools = data_bool
    all_values = data_value
#     all_bools = data_bool.cpu().detach().numpy()
#     all_values = data_value.cpu().detach().numpy()

    new_list = []
    for idx in range(len(all_bools)):
        new_list.append(tuple((all_bools[idx], all_values[idx])))
    return np.array(new_list, dtype='bool, i8')


class DataLoader(object):

    def __init__(self, fold=0, num_genes=500, mode='cpu'):
        self.fold = fold
        self.num_genes = num_genes
        self.load_data(mode)

    def load_data(self, mode):
        X_train_list, X_valid_list, y_value_train, y_value_valid, \
            y_train, y_valid = get_data(self.fold, config.csv_location, 'valid')
        _, X_test_list, _, y_value_test, _, y_test = \
            get_data(self.fold, config.csv_location, 'test')

        self.train_num = len(X_train_list)
        self.valid_num = len(X_valid_list)
        self.test_num = len(X_test_list)

        # labels

        if mode == 'cpu':
            self.y_train_bool = np.array(y_train)
            self.y_valid_bool = np.array(y_valid)
            self.y_test_bool = np.array(y_test)
            self.y_train_value = np.array(y_value_train)
            self.y_valid_value = np.array(y_value_valid)
            self.y_test_value = np.array(y_value_test)
        elif mode == 'gpu':
            from torch.autograd import Variable
            self.y_train_bool = Variable(torch.from_numpy(
                np.array(y_train))).float()
            self.y_valid_bool = Variable(torch.from_numpy(
                np.array(y_valid))).float()
            self.y_test_bool = Variable(torch.from_numpy(
                np.array(y_test))).float()
            self.y_train_value = Variable(torch.from_numpy(
                np.array(y_value_train))).float()
            self.y_valid_value = Variable(torch.from_numpy(
                np.array(y_value_valid))).float()
            self.y_test_value = Variable(torch.from_numpy(
                np.array(y_value_test))).float()
        else:
            raise(NotImplementedError)

        NRG = Dataset(config)

        # genomics
        X_gen_train, gen_list = NRG.get_genomics(X_train_list)
        X_gen_valid, gen_list = NRG.get_genomics(X_valid_list)
        X_gen_test, gen_list = NRG.get_genomics(X_test_list)

        all_std = np.std(np.array(X_gen_train), axis=0)
        all_sorted = np.argsort(all_std)
        X_gen_train = np.array(X_gen_train)[:, all_sorted[-self.num_genes:]]
        X_gen_valid = np.array(X_gen_valid)[:, all_sorted[-self.num_genes:]]
        X_gen_test = np.array(X_gen_test)[:, all_sorted[-self.num_genes:]]

        max_gen = np.max(np.concatenate(
             (X_gen_train, X_gen_valid, X_gen_train), axis=0))
        X_gen_train = (X_gen_train) / max_gen
        X_gen_valid = (X_gen_valid) / max_gen
        X_gen_test = (X_gen_test) / max_gen

        if mode == 'gpu':
            self.gen_train = Variable(torch.from_numpy(X_gen_train)).float()
            self.gen_valid = Variable(torch.from_numpy(X_gen_valid)).float()
            self.gen_test = Variable(torch.from_numpy(X_gen_test)).float()
        elif mode == 'cpu':
            self.gen_train = X_gen_train
            self.gen_valid = X_gen_valid
            self.gen_test = X_gen_test

        # pyradiomics
        X_pyrad_train = NRG.get_pyradiomics(X_train_list)
        X_pyrad_valid = NRG.get_pyradiomics(X_valid_list)
        X_pyrad_test = NRG.get_pyradiomics(X_test_list)

        max_pyrad = np.max(np.concatenate(
            (X_pyrad_train, X_pyrad_valid, X_pyrad_train), axis=0))
        X_pyrad_train = (X_pyrad_train) / max_pyrad
        X_pyrad_valid = (X_pyrad_valid) / max_pyrad
        X_pyrad_test = (X_pyrad_test) / max_pyrad

        if mode == 'gpu':
            self.pyrad_train = Variable(torch.from_numpy(X_pyrad_train)).float()
            self.pyrad_valid = Variable(torch.from_numpy(X_pyrad_valid)).float()
            self.pyrad_test = Variable(torch.from_numpy(X_pyrad_test)).float()
        elif mode == 'cpu':
            self.pyrad_train = X_pyrad_train
            self.pyrad_valid = X_pyrad_valid
            self.pyrad_test = X_pyrad_test

            
        # densenet
        X_dense_train = NRG.get_densenet_features(X_train_list)
        X_dense_valid = NRG.get_densenet_features(X_valid_list)
        X_dense_test = NRG.get_densenet_features(X_test_list)

        max_dense = np.max(np.concatenate(
            (X_dense_train, X_dense_valid, X_dense_train), axis=0))
        X_dense_train = (X_dense_train) / max_dense
        X_dense_valid = (X_dense_valid) / max_dense
        X_dense_test = (X_dense_test) / max_dense

        if mode == 'gpu':
            self.dense_train = Variable(torch.from_numpy(X_dense_train)).float()
            self.dense_valid = Variable(torch.from_numpy(X_dense_valid)).float()
            self.dense_test = Variable(torch.from_numpy(X_dense_test)).float()
        elif mode == 'cpu':
            self.dense_train = X_dense_train
            self.dense_valid = X_dense_valid
            self.dense_test = X_dense_test


    