# IMPACT_NMR

IMPACT (Intelligent Machine Predicting Accurate Chemical Tensors) is a kernel ridge regression machine learning model for prediction of chemical tensors at the wb97xd | 6-311g(d,p) level of theory. Developed with datasets of molecules from the Cambridge Structural Database, IMPACT takes as an input log files with results of chemical tensors at cheap wb97xd | 3-21G level of theory. Predicted tensors are then scaled using scaling factors derived using the CHESHIRE datasets.
This project was developed by Krystof Chrappova as part of Msci Chemistry final year project at the University of Bristol.


## Dependencies
* scikit-learn              0.24.2
* scikit-optimize           0.9.0
* dscribe                   1.1.0
* ase                       3.22.0
* pandas                    1.3.2
* numpy                     1.21.2

## Usage
IMPACT works as follows:
1. Run Gaussian calculation with wb97xd | 3-21G level of theory and create `.sdf` files from the `.log` files using mol_translator without scaling the tensors. Create atoms dataframe of the results using mol translator.
5. using make_delta_inp.py prepare the input for IMPACT.
6. using delta_predict_H.py / delta_predict_C.py IMPACT predicts wb97xd | 6-311g(d,p) level of theory chemical tensors that are scaled afterwards in the IMPACT output file.


## Retraining
If you wish to retrain the model you can do so using train_test_IMPACT_C.py / train_test_IMPACT_H.py. Here you need to change the path to high level and low-level calculation results.


## Known limitations:
Molecular size - any molecule that has more than 80 features created with ACSF representation will raise an error. 
