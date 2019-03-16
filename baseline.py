import pandas
import numpy as np

# Load data from warfarin.csv
df = pandas.read_csv('warfarin.csv', header = None)
data = np.array(df)[1:-1, :-3]
titles = np.array(df)[0, :-3]

# Removing patients with unknown dosage
true_labels = data[:, 34].astype(np.float)
valid_labels = np.where(np.isfinite(true_labels))[0]
valid_data = data[valid_labels]
num_patients, num_features = valid_data.shape

# Processing true dosage
low = 21.0
high = 49.0
dosage = valid_data[:, 34].astype(np.float)

# Function which converts continuous dosage values to discretized values (0.0, 0.5, 0.1)
def dosage_discretize(dosage):
	discrete_dosage = []
	for dose in dosage.tolist():
		if dose < low:
			discrete_dosage.append(0.0)
		elif dose <= high:
			discrete_dosage.append(0.5)
		else:
			discrete_dosage.append(1.0)
	discrete_dosage = np.array(discrete_dosage)
	return discrete_dosage

right_dosage = dosage_discretize(dosage)

# Fixed-dose
pf_fixed = np.count_nonzero(right_dosage == 0.5)/num_patients

# Warfarin clinical dosing algorithm
age_decades = []
height_cm = []
weight_kg = []
asian_race = []
black_afro = []
missing_mixed = []
enzyme = []
amiodarone = []
true_dosage = []
n_patient = 0

for patient, dosage in zip(valid_data.tolist(), right_dosage.tolist()):
	if type(patient[4]) == str and type(patient[5]) == str and type(patient[6]) == str and \
	type(patient[2]) == str and type(patient[24]) == str and type(patient[25]) == str and \
	type(patient[26]) == str and type(patient[23]) == str:
		n_patient += 1
		true_dosage.append(dosage)
		age_decades.append(float(patient[4][0]))
		height_cm.append(float(patient[5]))
		weight_kg.append(float(patient[6]))
		if patient[2] == 'Asian':
			asian_race.append(1.0)
		else:
			asian_race.append(0.0)
		if patient[2] == 'Black or African American':
			black_afro.append(1.0)
		else:
			black_afro.append(0.0)	
		if patient[2] == 'Unknown':
			missing_mixed.append(1.0)
		else:
			missing_mixed.append(0.0)
		if patient[24] == 1 and patient[25] == 1 and patient[26] == 1:	
			enzyme.append(1.0)
		else:
			enzyme.append(0.0)
		if patient[23] == 1:
			amiodarone.append(1.0)
		else:
			amiodarone.append(0.0)

age_decades = np.array(age_decades)
height_cm = np.array(height_cm)
weight_kg = np.array(weight_kg)
asian_race = np.array(asian_race)
black_afro = np.array(black_afro)
missing_mixed = np.array(missing_mixed)
enzyme = np.array(enzyme)
amiodarone = np.array(amiodarone)
true_dosage = np.array(true_dosage)

prediction = np.square(4.0376 - 0.2546*age_decades + 0.0118*height_cm + 0.0134*weight_kg - \
0.6752*asian_race + 0.4060*black_afro + 0.0443*missing_mixed + 1.2799*enzyme - 0.5695*amiodarone)
predicted_dosage = dosage_discretize(prediction)
pf_clinical = np.count_nonzero(predicted_dosage == true_dosage)/n_patient




