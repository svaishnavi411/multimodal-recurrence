'''
Contains the utility functions
'''

from sklearn.preprocessing import MinMaxScaler
import numpy as np
import config as config

import sys
sys.path.append('../')

from loader import Dataset
from fusion_functions import get_data
import torch
from torch.autograd import Variable



available_models = ['genomics', 'pyradiomics',
                    'densenet', 'intermediate_gp',
                    'intermediate_gd', 'late_gp',
                    'late_gd']


def get_structured_array(data_bool, data_value):
    all_bools = data_bool.cpu().detach().numpy()
    all_values = data_value.cpu().detach().numpy()

    new_list = []
    for idx in range(len(all_bools)):
        new_list.append(tuple((all_bools[idx], all_values[idx])))
    return np.array(new_list, dtype='bool, i8')


class DataLoader(object):

    def __init__(self, fold=0, num_genes=500):
        self.fold = fold
        self.num_genes = num_genes
        self.load_data()

    def load_data(self):
        X_train_list, X_valid_list, y_value_train, y_value_valid, \
            y_train, y_valid = get_data(self.fold, config.csv_location, 'valid')
        _, X_test_list, _, y_value_test, _, y_test = \
            get_data(self.fold, config.csv_location, 'test')

        self.train_num = len(X_train_list)
        self.valid_num = len(X_valid_list)
        self.test_num = len(X_test_list)

        # labels
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

#         scaler = MinMaxScaler()
#         scaler.fit(np.concatenate(
#             (X_gen_train, X_gen_valid, X_gen_train), axis=0))
        max_gen = np.max(np.concatenate(
             (X_gen_train, X_gen_valid, X_gen_train), axis=0))
        X_gen_train = (X_gen_train) / max_gen
        X_gen_valid = (X_gen_valid) / max_gen
        X_gen_test = (X_gen_test) / max_gen

        self.gen_train = Variable(torch.from_numpy(X_gen_train)).float()
        self.gen_valid = Variable(torch.from_numpy(X_gen_valid)).float()
        self.gen_test = Variable(torch.from_numpy(X_gen_test)).float()

        # pyradiomics
        X_pyrad_train = NRG.get_pyradiomics(X_train_list)
        X_pyrad_valid = NRG.get_pyradiomics(X_valid_list)
        X_pyrad_test = NRG.get_pyradiomics(X_test_list)

#         scaler = MinMaxScaler()
#         scaler.fit(np.concatenate(
#             (X_pyrad_train, X_pyrad_valid, X_pyrad_train), axis=0))

        max_pyrad = np.max(np.concatenate(
            (X_pyrad_train, X_pyrad_valid, X_pyrad_train), axis=0))
        X_pyrad_train = (X_pyrad_train) / max_pyrad
        X_pyrad_valid = (X_pyrad_valid) / max_pyrad
        X_pyrad_test = (X_pyrad_test) / max_pyrad

        self.pyrad_train = Variable(torch.from_numpy(X_pyrad_train)).float()
        self.pyrad_valid = Variable(torch.from_numpy(X_pyrad_valid)).float()
        self.pyrad_test = Variable(torch.from_numpy(X_pyrad_test)).float()

        # densenet
        X_dense_train = NRG.get_densenet_features(X_train_list)
        X_dense_valid = NRG.get_densenet_features(X_valid_list)
        X_dense_test = NRG.get_densenet_features(X_test_list)

#         scaler = MinMaxScaler()
        max_dense = np.max(np.concatenate(
            (X_dense_train, X_dense_valid, X_dense_train), axis=0))
        X_dense_train = (X_dense_train) / max_dense
        X_dense_valid = (X_dense_valid) / max_dense
        X_dense_test = (X_dense_test) / max_dense

        self.dense_train = Variable(torch.from_numpy(X_dense_train)).float()
        self.dense_valid = Variable(torch.from_numpy(X_dense_valid)).float()
        self.dense_test = Variable(torch.from_numpy(X_dense_test)).float()

    def data_loader(mode, train=True, fold=0):
        return




