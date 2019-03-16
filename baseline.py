import pandas
import numpy as np

#Load data from warfarin.csv
df = pandas.read_csv('warfarin.csv', header = None)
data = np.array(df)[1:-1, :-3]
titles = np.array(df)[0, :-3]

#Removing patients with unknown dosage
true_labels = data[:, 34].astype(np.float)
valid_labels = np.where(np.isfinite(true_labels))[0]
valid_data = data[valid_labels]
num_patients, num_features = valid_data.shape
dosage = valid_data[:, 34].astype(np.float)

#Fixed dose baseline
fixed_dose = 35.0
pf_fixed = np.count_nonzero(dosage == fixed_dose)/num_patients

