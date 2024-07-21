# ESATED

ESATED: leveraging Extra-weak Supervision with Auxiliary Task for Enhanced non-intrusiveness in Energy Disaggregation



We use anaconda and pytorch in a win11 environment. The environment file is 'endis.yaml.'

This repository consists of 4 folders, as.

1. dataset_management: It contains codes for generate processed data from public datasets. 
2. data_process: It contains all data processing functions. 
3. model: It contains model designs and functions which are used for training and evaluating.
4. experiment: It has two folders within. 'overall_exp' is for overall experiments for public datasets and our self-collected dataset. 'sub_exp' is for all other experiments such as ablation experiments. 

We prefer using Jupiter notebook, so experiment codes are mostly *.ipynb files. 

The self-collected dataset is too big to be uploaded to github, so we store it on google drive, at https://drive.google.com/file/d/1ZFf-hjUMrFqHqP-mxPFwZ_nJgWysUfWd/view?usp=drive_link and https://drive.google.com/file/d/1m7mPsoojRqQ6rwQecvQFTqF4PWR_vw0I/view?usp=sharing.

