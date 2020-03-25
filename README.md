# multimodal-recurrence
Code accompanying our ISBI 2020 paper 
[Multimodal fusion of imaging and genomics for lung cancer recurrence prediction](https://arxiv.org/pdf/2002.01982.pdf)
by Vaishnavi Subramanian, Minh N. Do, Tanveer Syeda-Mahmood

#### Data
The original dataset can be downloaded from the TCIA repository [here](https://wiki.cancerimagingarchive.net/display/Public/NSCLC+Radiogenomics).

We also provide a curated dataset containing the images and segmentations we generated and used. The segmentations were originally available in the XML format, which we have converted to NIFTI files of the same size as the original CT image for easy re-use. We also provide the splits we used in our experiments for use in comparison studies. 

The curated dataset used in our experiments can be downloaded from [here](https://uofi.box.com/v/nsclc-radiogenomics-curated). This dataset contains

- ``images``: The studies from the original dataset which were used to generate the original segmentations in the folder
- ``segmentations``: The segmentations converted to NIFTI files from the AIM annotaion XMLs provided in the original dataset
- ``full_clinical_file.csv``: The original CSV file contain all the clinical information
- ``recurrence_splits``: The folder contain all the 5 splits we used in our experiments. Each split file contains the data in rows where each row corresponds to patient_ID | num_of_days | recurrence_bool. The patient_ID is the same as that provided in the original dataset. The value recurrence_bool is 1 if the patient's cancer recurred within the study period, and 0 otherwise. The num_of_days refers to the number of days between the CT image and the recurrence, if the corresponding recurrence_bool is 1. The num_of_days refers to the number of days between the CT image and the day the patient was last tracked in the study. 

#### Code
Accompanying code will be posted here during the ISBI conference. Stay tuned!
