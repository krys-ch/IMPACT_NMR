import pandas as pd
import numpy as np
import pickle
tensor_ll = pd.read_csv(r'...')
zeros = []

for element in np.arange((len(tensor_ll.columns)-12), 80, 1).tolist():
    a = str(element)
    zeros.append(a)
tensor_ll[zeros] = 0


features = np.arange(0, 80, 1).tolist()
features.append('shift')
f = []
for x in features:
    a = str(x)
    f.append(a)
labels = ['shift']


model = pickle.load(open('IMPACT_H.pkl', 'rb'))
tensor_hl = np.array(model.predict(tensor_ll[f]).tolist())

tensor_ll['shift_low_level'] = (tensor_ll['shift'] - 32.9843) / -1.0205
tensor_ll['IMPACT_shift'] = (tensor_hl -32.1254)/-1.0719

result_df = tensor_ll[['molecule_name', 'atom_index', 'typeint', 'conn', 'x', 'y', 'z','shift_low_level', 'IMPACT_shift']]
result_df.to_csv('IMPACT_prediction_strych_H_testing.csv')
