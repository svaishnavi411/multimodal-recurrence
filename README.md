# multimodal-recurrence
Code accompanying the paper 
"MULTIMODAL FUSION OF IMAGING AND GENOMICS FOR LUNG CANCER RECURRENCE PREDICTION"
- Vaishnavi Subramanian, Minh N. Do, Tanveer Syeda-Mahmood (ISBI 2020)

#### Data
The data splits used in our experiments can be found in ```data_splits/```
The original dataset can be downloaded at https://wiki.cancerimagingarchive.net/display/Public/NSCLC+Radiogenomics 

The curated dataset used in our experiments can be downloaded from: 

#### Linear Models

-> can be run from ```linear_cox.ipynb```

#### Neural Net Models

-> supporting files [net.py, utils.py etc..]
-> train.py [the file called to train models]
-> run.py - run as needed - important
-> clean_results.ipynb -> run to pick the best hyperparameters for each model
-> test.py - run to test models

#### TD-AUCs
have a different notebook for this:
- generate_tdaucs.ipynb