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

def dosage_discretize(dosage):
	"""
	Converts continuous dosage values to discretized values (0.0, 0.5, 0.1)

	@param dosage (numpy array): 1 dimensional numpy array or list of continuous dosage values
	@returns discrete_dosage (numpy array): 1 dimensional numpy array of discrete dosage values

	"""
	discrete_dosage = []
	for dose in dosage:
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

# Imputation of VKORC1 SNPs
for patient in valid_data:
	if type(patient[41]) == float:
		if patient[2] != 'Black or African American' or patient[2] != 'Unknown' and patient[51] == 'C/C':
			patient[41] = 'G/G'
		if patient[2] != 'Black or African American' or patient[2] != 'Unknown' and patient[51] == 'T/T':
			patient[41] = 'A/A'
		if patient[2] != 'Black or African American' or patient[2] != 'Unknown' and patient[51] == 'C/T':
			patient[41] = 'A/G'
		if patient[45] == 'C/C':
			patient[41] = 'G/G'
		if patient[45] == 'T/T':
			patient[41] = 'A/A'
		if patient[45] == 'C/T':
			patient[41] = 'A/G'	
		if patient[2] != 'Black or African American' or patient[2] != 'Unknown' and patient[47] == 'G/G':
			patient[41] = 'G/G'
		if patient[2] != 'Black or African American' or patient[2] != 'Unknown' and patient[47] == 'C/C':
			patient[41] = 'A/A'
		if patient[2] != 'Black or African American' or patient[2] != 'Unknown' and patient[47] == 'C/G':
			patient[41] = 'A/G'			

# Warfarin Clinical Dosing Algorithm and Warfarin Pharmacogenetic Dosing Algorithm
age_decades = []
height_cm = []
weight_kg = []
heterozygous = []
homozygous = []
type_unknown = []
cyp12 = []
cyp13 = []
cyp22 = []
cyp23 = []
cyp33 = []
cyp_unknown = []
asian_race = []
black_afro = []
missing_mixed = []
enzyme = []
amiodarone = []
true_dosage = []
n_patient = 0

for patient, dosage in zip(valid_data, right_dosage):
	if type(patient[4]) == str and type(patient[5]) == str and type(patient[6]) == str and \
	type(patient[2]) == str and type(patient[24]) == str and type(patient[25]) == str and \
	type(patient[26]) == str and type(patient[23]) == str:
		n_patient += 1
		true_dosage.append(dosage)
		age_decades.append(float(patient[4][0]))
		height_cm.append(float(patient[5]))
		weight_kg.append(float(patient[6]))
		if patient[41] == 'A/G':
			heterozygous.append(1.0)
		else:
			heterozygous.append(0.0)
		if patient[41] == 'A/A':
			homozygous.append(1.0)
		else:
			homozygous.append(0.0)
		if type(patient[41]) == float:
			type_unknown.append(1.0)
		else:
			type_unknown.append(0.0)
		if patient[37] == '*1/*2':
			cyp12.append(1.0)
		else:
			cyp12.append(0.0)
		if patient[37] == '*1/*3':
			cyp13.append(1.0)
		else:
			cyp13.append(0.0)
		if patient[37] == '*2/*2':
			cyp22.append(1.0)
		else:
			cyp22.append(0.0)
		if patient[37] == '*2/*3':
			cyp23.append(1.0)
		else:
			cyp23.append(0.0)
		if patient[37] == '*3/*3':
			cyp33.append(1.0)
		else:
			cyp33.append(0.0)
		if type(patient[37]) == float:
			cyp_unknown.append(1.0)
		else:
			cyp_unknown.append(0.0)									
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
heterozygous = np.array(heterozygous)
homozygous = np.array(homozygous)
type_unknown = np.array(type_unknown)
cyp12 = np.array(cyp12)
cyp13 = np.array(cyp13)
cyp22 = np.array(cyp22)
cyp23 = np.array(cyp23)
cyp33 = np.array(cyp33)
cyp_unknown = np.array(cyp_unknown)
asian_race = np.array(asian_race)
black_afro = np.array(black_afro)
missing_mixed = np.array(missing_mixed)
enzyme = np.array(enzyme)
amiodarone = np.array(amiodarone)
true_dosage = np.array(true_dosage)

prediction_cd = np.square(4.0376 - 0.2546*age_decades + 0.0118*height_cm + 0.0134*weight_kg - \
0.6752*asian_race + 0.4060*black_afro + 0.0443*missing_mixed + 1.2799*enzyme - 0.5695*amiodarone)
cd_dosage = dosage_discretize(prediction_cd)
pf_clinical = np.count_nonzero(cd_dosage == true_dosage)/n_patient

prediction_pd = np.square(5.6044 - 0.2614*age_decades + 0.0087*height_cm + 0.0128*weight_kg - 0.8677*heterozygous - 1.6974*homozygous\
- 0.4854*type_unknown - 0.5211*cyp12 - 0.9357*cyp13 - 1.0616*cyp22 - 1.9206*cyp23 - 2.3312*cyp33 - 0.2188*cyp_unknown\
- 0.1092*asian_race - 0.2760*black_afro - 0.1032*missing_mixed + 1.1816*enzyme - 0.5503*amiodarone)
pd_dosage = dosage_discretize(prediction_pd)
pf_pharma = np.count_nonzero(pd_dosage == true_dosage)/n_patient


# Feature engineering

# Median age, average height, average weight
age_count = 0
ages = []
height_count = 0
height_sum = 0
weight_count = 0
weight_sum = 0

for patient in valid_data:
	if type(patient[4]) == str:
		age_count += 1
		ages.append(float(patient[4][0]))
	if type(patient[5]) == str:
		height_count += 1
		height_sum += float(patient[5])
	if type(patient[6]) == str:
		weight_count += 1
		weight_sum += float(patient[6])

ages = sorted(ages)
median_age = int((ages[age_count//2 - 1] + ages[age_count//2])/2)
height_avg = height_sum/height_count
weight_avg = weight_sum/weight_count

gender = []
race = []
ethnicity = []
age_decades = []
height_cm = []
weight_kg = []
cyp2 = []
vk3673 = []
vk5808 = []
vk6484 = []
vk6853 = []
vk9041 = []
vk7566 = []
vk961 = []
bias = []

def one_hot(feature, str1, str2, query):
	if type(query) == float:
		feature.append([0.0, 0.0, 0.0, 1.0])
	elif query == str1:
		feature.append([1.0, 0.0, 0.0, 0.0])
	elif query == str2:
		feature.append([0.0, 1.0, 0.0, 0.0])
	else:
		feature.append([0.0, 0.0, 1.0, 0.0])

for patient in valid_data:
	if type(patient[1]) == float:
		gender.append([0.0, 0.0, 1.0])
	elif patient[1] == 'male':
		gender.append([1.0, 0.0, 0.0])
	else:
		gender.append([0.0, 1.0, 0.0])
	if patient[2] == 'Asian':
		race.append([1.0, 0.0, 0.0, 0.0])
	elif patient[2] == 'White':
		race.append([0.0, 1.0, 0.0, 0.0])
	elif patient[2] == 'Black or African American':
		race.append([0.0, 0.0, 1.0, 0.0])
	else:
		race.append([0.0, 0.0, 0.0, 1.0])
	if patient[3] == 'Hispanic or Latino':
		ethnicity.append([1.0, 0.0, 0.0])
	elif patient[3] == 'not Hispanic or Latino':
		ethnicity.append([0.0, 1.0, 0.0])
	else:
		ethnicity.append([0.0, 0.0, 1.0])
	if type(patient[4]) == float:
		age_decades.append(median_age - 1)
	else:
		age_decades.append(int(patient[4][0]) - 1)
	if type(patient[5]) == float:
		height_cm.append(height_avg)
	else:
		height_cm.append(float(patient[5]))
	if type(patient[6]) == float:
		weight_kg.append(weight_avg)
	else:
		weight_kg.append(float(patient[6]))
	if patient[37] == '*1/*1':
		cyp2.append([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
	elif patient[37] == '*1/*2':
		cyp2.append([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
	elif patient[37] == '*1/*3':
		cyp2.append([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
	elif patient[37] == '*2/*2':
		cyp2.append([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])		
	elif patient[37] == '*2/*3':
		cyp2.append([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
	elif patient[37] == '*3/*3':
		cyp2.append([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
	else:
		cyp2.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
	one_hot(vk3673, 'A/A', 'A/G', patient[41])
	one_hot(vk5808, 'G/G', 'G/T', patient[43])
	one_hot(vk6484, 'C/C', 'C/T', patient[45])
	one_hot(vk6853, 'C/C', 'C/G', patient[47])
	one_hot(vk9041, 'A/A', 'A/G', patient[49])
	one_hot(vk7566, 'C/C', 'C/T', patient[51])
	one_hot(vk961, 'A/A', 'A/C', patient[53])
	bias.append([1.0])

age_decades = np.array(age_decades)
age_bins = 9
age_features = np.zeros((num_patients, age_bins))
age_features[np.arange(num_patients), age_decades] = 1.0

height_bins = 8
height_discrete = np.array([min(int((height - 120)/10), height_bins - 1) for height in height_cm])
height_features = np.zeros((num_patients, height_bins))
height_features[np.arange(num_patients), height_discrete] = 1.0

weight_bins = 10
weight_discrete = np.array([min(int((weight - 30)/20), weight_bins - 1) for weight in weight_kg])
weight_features = np.zeros((num_patients, weight_bins))
weight_features[np.arange(num_patients), weight_discrete] = 1.0

features = np.array([g + r + e + c + v1 + v2 + v3 + v4 + v5 + v6 + v7 + b for g, r, e, c, v1, v2, v3, v4, v5, v6, v7, b in \
zip(gender, race, ethnicity, cyp2, vk3673, vk5808, vk6484, vk6853, vk9041, vk7566, vk961, bias)])

features = np.column_stack((features, age_features, height_features, weight_features))























