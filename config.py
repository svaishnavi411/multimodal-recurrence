# LOCATION VARIABLES

#TODO: Ensure that these can be run from other machines easily.
location = 'data/stanford/'

clinical = location + 'labels/full_clinical_file.csv'
recurrence = location + 'labels/recurrence_cleaned.csv'
images = location + 'images_original/'
segs = location + 'segmentations_cleaned/'
cropped = location + 'cropped_nodules/'
pyradiomics = location + 'pyradiomics_features/'
genomics = location + 'genomics_cleaned.csv'
densenet = cropped
csv_location = location + 'labels/recurrence_labels/'

# OTHER VARIABLES

mutation_mapping = {'Mutant': 1,
                    'Wildtype': 0,
                    'Not collected': 'N/A',
                    'Unknown': 'N/A'}

recurrence_mapping = {'yes': 1, 'no': 0, 'Not collected': 'N/A'}
survival_mapping = {'Dead': 1, 'Alive': 0}

class_weights = {'Recurrence': {'no':0.3, 'yes':0.7},
                 'Survival Status': {'Dead':0.8, 'Alive':0.2}}
location_dict = {'Checked': 1, 'Unchecked':0}
smoking_dict = {'Nonsmoker': 0,
                'Former' : 1,
                'Current' : 2}
gg_dict = {'0%': 0,
           '>0 - 25%': 1,
           '25 - 50%': 2,
           '50 - 75%' : 3,
           '75 - < 100%' : 4,
           '100%' : 5,
           'Not Assessed': 0}
