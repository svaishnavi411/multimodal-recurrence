'''
This file contains a class which will load the different
attributes of the patients of a given dataset.
Firstly, NSCLC-Radiogenomics
'''

from copy import deepcopy
import pandas as pd
import numpy as np
import os
from itertools import chain
from sklearn.preprocessing import LabelBinarizer
import nibabel as nib


class Dataset(object):
    '''
    Class which contains a list of all patients,
    images, genomics, recurrence, survival, and mutation information.

    If the corresponding data is unavailable, 'NA' will
    be used.
    '''

    def __init__(self, config, dataset='NSCLC-Radiogenomics'):
        self.dataset_info = dataset
        self.config = config
        self.get_patient_list()

        # these are list of files which should be read appropriately
        self.image_list = {}
        self.images = {}
        self.seg_list = {}
        self.feature_list = {}

        # these are already loaded attributes
        self.genomics_list = {}
        self.egfr_mutation = {}
        self.recurrence_bool = {}
        self.recurrence_value = {}
        self.survival_bool = {}
        self.survival_value = {}
        self.durations = {}

        #TODO: include this:       self.last_known_alive = []
        self.clinical_list = {}

        self.load_all()

    def get_patient_list(self):

        self.data_location = self.config.location
        self.patient_list = list(pd.read_csv(self.config.clinical)['Case ID'])
        return

    def set_patient_list(self, patient_list):
        self.patient_list = patient_list

    def load_all(self):
        self.load_images()
        self.load_segmentations()
        self.load_pyradiomics()
        self.load_genomics()
        self.load_recurrence()
        self.load_survival()
        self.load_egfr_mutation()
#         self.load_clinical()
        self.load_densenet_features()

    def load_images(self):
        for patientID in self.patient_list:
            image_path = self.config.images + patientID + '.nii'
            if os.path.exists(image_path):
                self.image_list[patientID] = image_path
            else:
                self.image_list[patientID] = 'N/A'
        return

    def get_images_cropped(self):
        self.images = {}
        for patientID in self.patient_list:
            image_path = self.config.cropped + patientID + '_cropped_nodule.nii'
            if os.path.exists(image_path):
                self.images[patientID] = nib.load(image_path).get_fdata()
            else:
                self.images[patientID] = 'N/A'
        return

    def get_pyradiomics(self, patient_list):
        features=[]
        for patientID in patient_list:
            feature_path = self.config.pyradiomics + patientID + '_dilated.npz'
            features.append(np.load(feature_path)['arr_0'])
        return features

    def get_densenet_features(self, patient_list):
        features=[]
        for patientID in patient_list:
            feature_path = self.config.densenet + patientID + '_densenet.npy'
            loaded = np.load(feature_path)
            features.append(loaded)

        features = np.array(features)
        features = np.squeeze(features)
        return features

    def get_genomics(self, patient_list):
        genomics = pd.read_csv(self.config.genomics, index_col=False)
        genomics.set_index('Unnamed: 0.1', inplace=True)
        genomics = genomics.drop('Unnamed: 0', axis=1)
        genomics = genomics.transpose()

        # TODO: we can add code here to normalize the genomics data
        genomics_list = []
        for id in patient_list:
            genomics_list.append(list(genomics.loc[id]))
        return genomics_list, genomics.columns


    def get_clinical(self, patient_list):
        clinical = pd.read_csv(self.config.clinical)

        list_of_variables = clinical.columns.values

        predictors_labels = list(chain([list_of_variables[0]], [list_of_variables[2]], list_of_variables[6:8], list_of_variables[9:22], list_of_variables[23:24])) #:30]))
        predictors = clinical[predictors_labels]

        predictors.set_index('Case ID', inplace=True)

        predictors['Smoking status'].replace(self.config.smoking_dict, inplace=True)
        predictors['Pack Years'].replace({'N/A': 0, 'Not Collected': 40}, inplace=True)
        predictors['%GG'].replace(self.config.gg_dict, inplace=True)
        for idx in range(10, 17):
            predictors[list_of_variables[idx]].replace(self.config.location_dict, inplace=True)

        encoder = LabelBinarizer()
        for idx in range(17, 24):
            if idx == 22:
                continue
            predictors[list_of_variables[idx]] = encoder.fit_transform(predictors[list_of_variables[idx]])
        predictors.fillna(0, inplace=True)

        clinical_data = []

        for id in patient_list:
            clinical_data.append([int(x) for x in list(predictors.loc[id])])

        return clinical_data

    def load_segmentations(self):
        self.seg_list = {}
        for patientID in self.patient_list:
            seg_path = self.config.segs + patientID + '.nii.gz'
            if os.path.exists(seg_path):
                self.seg_list[patientID] = seg_path
            else:
                self.seg_list[patientID] = 'N/A'
        return

    def load_pyradiomics(self):

        self.feature_list = {}
        for patientID in self.patient_list:
            feature_path = self.config.pyradiomics + patientID + '.npz'
            if os.path.exists(feature_path):
                self.feature_list[patientID] = feature_path
            else:
                self.feature_list[patientID] = 'N/A'
        return

    def load_densenet_features(self):
        self.densenet_features = {}
        for patientID in self.patient_list:
            feature_path = self.config.densenet + patientID + '_densenet.npy'
            if os.path.exists(feature_path):
                temp = np.load(feature_path)
                if np.size(temp) == 1:
                    self.densenet_features[patientID] = 'N/A'
                else:
                    self.densenet_features[patientID] = feature_path
            else:
                self.densenet_features[patientID] = 'N/A'
        return

    def load_genomics(self):
        self.genomics_list = {}
        genomics = pd.read_csv(self.config.genomics, index_col=False)
        genomics.set_index('Unnamed: 0.1', inplace=True)
        genomics = genomics.drop('Unnamed: 0', axis=1)
        genomics = genomics.transpose()

        for id in self.patient_list:
            if id in genomics.index.values:
               self.genomics_list[id] = list(genomics.loc[id])
            else:
                self.genomics_list[id] = 'N/A'
        return

    def load_recurrence(self):
        #TODO: Include the location information as well

        self.recurrence_value = {}
        self.recurrence_bool = {}
        self.durations = {}
        recurrence = pd.read_csv(self.config.recurrence, index_col=False)
        recurrence.set_index('Case ID', inplace=True)

        for id in self.patient_list:
            if id in recurrence.index.values:
                curr_patient = recurrence.loc[id]
                value = curr_patient['Recurrence']
                self.recurrence_bool[id] = value
                self.recurrence_value[id] = curr_patient['Days']

        return

    def load_survival(self):

        self.survival_value = {}
        self.survival_bool = {}
        recurrence = pd.read_csv(self.config.clinical, index_col=False)
        recurrence.set_index('Case ID', inplace=True)

        for id in self.patient_list:
            if id in recurrence.index.values:
                curr_patient = recurrence.loc[id]
                value = curr_patient['Survival Status']
                mapped_value = self.config.survival_mapping[value]
                self.survival_bool[id] = mapped_value

                if mapped_value == 1:
                    self.survival_value[id] = curr_patient['Time to Death (days)']
                else:
                    self.survival_value[id] = 'N/A'
            else:
                self.survival_bool[id] = 'N/A'
                self.survival_value[id] = 'N/A'
        return

    def load_egfr_mutation(self):
        self.egfr_mutation = {}
        egfr = pd.read_csv(self.config.clinical, index_col=False)

        egfr.set_index('Case ID', inplace=True)

        for id in self.patient_list:
            if id in egfr.index.values:
                value = egfr.loc[id]['EGFR mutation status']
                mapped_value = self.config.mutation_mapping[value]
                self.egfr_mutation[id] = mapped_value

            else:
                self.egfr_mutation[id] = 'N/A'
        return

    def load_clinical(self):
        self.clinical_list = {}
        clinical = pd.read_csv(self.config.clinical)

        list_of_variables = clinical.columns.values

        # clinical = clinical.loc[49:]

        # for id in range(len(list_of_variables)):
        #     print(id, list_of_variables[id])

        predictors_labels = list(chain([list_of_variables[0]], [list_of_variables[2]], list_of_variables[6:8], list_of_variables[9:22], list_of_variables[23:24])) #:30]))
        predictors = clinical[predictors_labels]

        predictors.set_index('Case ID', inplace=True)

        predictors['Smoking status'].replace(self.config.smoking_dict, inplace=True)
        predictors['Pack Years'].replace({'N/A': 0, 'Not Collected': 40}, inplace=True)
        predictors['%GG'].replace(self.config.gg_dict, inplace=True)
        for idx in range(10, 17):
            predictors[list_of_variables[idx]].replace(self.config.location_dict, inplace=True)

        encoder = LabelBinarizer()
        for idx in range(17, 24):
            if idx == 22:
                continue
            predictors[list_of_variables[idx]] = encoder.fit_transform(predictors[list_of_variables[idx]])
        predictors.fillna(0, inplace=True)

        print(predictors.columns.values)
        for id in self.patient_list:
            self.clinical_list[id] = [int(x) for x in list(predictors.loc[id])]
        return

    def select_subset_patients(self, to_select, replace_list=False):
        '''

        :param dataset: dataset of type Dataset
        :param to_select: list of features to subselect
        :return: updated dataset
        '''

        patient_list = deepcopy(self.patient_list)
        for id in self.patient_list:
            remove_bool = False
            for attr in to_select:
                if attr == 'pyradiomics':
                    if self.feature_list[id] == 'N/A':
                        remove_bool = True
                if attr == 'gene_expressions':
                    if self.genomics_list[id] == 'N/A':
                        remove_bool = True
                if attr == 'clinical':
                    if self.clinical_list[id] == 'N/A':
                        remove_bool = True
                if attr == 'recurrence':
                    if self.recurrence_bool[id] == 'N/A':
                        remove_bool = True
                if attr == 'densenet':
                    if self.densenet_features[id] == 'N/A':
                        remove_bool = True
                if attr == 'egfr':
                    if self.egfr_mutation[id] == 'N/A':
                        remove_bool = True
            if remove_bool is True:
                patient_list.remove(id)

        if replace_list == True:
            self.set_patient_list(patient_list)
            self.load_all()

        return patient_list

if __name__ == '__main__':
    nrg = Dataset()
