import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt

# Load data from warfarin.csv
df = pd.read_csv('warfarin.csv', header = None)
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
	Converts continuous dosage values to discretized values (0, 1, 2)

	@param dosage (numpy array): 1 dimensional numpy array or list of continuous dosage values
	@returns discrete_dosage (numpy array): 1 dimensional numpy array of discrete dosage values

	"""
	discrete_dosage = []
	for dose in dosage:
		if dose < low:
			discrete_dosage.append(0)
		elif dose <= high:
			discrete_dosage.append(1)
		else:
			discrete_dosage.append(2)
	discrete_dosage = np.array(discrete_dosage)
	return discrete_dosage

right_dosage = dosage_discretize(dosage)

# Fixed-dose
pf_fixed = np.count_nonzero(right_dosage == 1)/num_patients

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

# print (pf_fixed, pf_clinical, pf_pharma)

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
diagnosis = []
diabetes = []
heart = []
valve = []
smoker = []
medic = [[] for i in range(18)]
cyp2 = []
vk3673 = []
vk5808 = []
vk6484 = []
vk6853 = []
vk9041 = []
vk7566 = []
vk961 = []
bias = []

def vkorc1(feature, str1, str2, query):
	if type(query) == float:
		feature.append([0.0, 0.0, 0.0, 1.0])
	elif query == str1:
		feature.append([1.0, 0.0, 0.0, 0.0])
	elif query == str2:
		feature.append([0.0, 1.0, 0.0, 0.0])
	else:
		feature.append([0.0, 0.0, 1.0, 0.0])

def three_hot(feature, index):
	if type(patient[index]) == float:
		feature.append([0.0, 0.0, 1.0])
	elif patient[index] == '1':
		feature.append([1.0, 0.0, 0.0])
	else:
		feature.append([0.0, 1.0, 0.0])

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
		age_decades.append([(median_age - 1)/8])
	else:
		age_decades.append([(float(patient[4][0]) - 1)/8])
	if type(patient[5]) == float:
		height_cm.append(height_avg)
	else:
		height_cm.append(float(patient[5]))
	if type(patient[6]) == float:
		weight_kg.append(weight_avg)
	else:
		weight_kg.append(float(patient[6]))
	if type(patient[7]) == float:
		array = np.zeros(9)
		array[0] = 1.0
		diagnosis.append(array.tolist())
	else:
		array = np.zeros(9)
		lis = [int(c) for c in list(patient[7]) if c.isdigit()]
		array[lis] = 1.0
		diagnosis.append(array.tolist())
	three_hot(diabetes, 9)
	three_hot(heart, 10)
	three_hot(valve, 11)
	three_hot(smoker, 36)
	for i in range(13, 31):
		if i != 22:
			three_hot(medic[i-13], i)  
		else:
			if type(patient[i]) == float:
				medic[i-13].append([0.0, 1.0])
			else:
				medic[i-13].append([1.0, 0.0])
	if patient[37] == '*1/*1':
		array = np.zeros(12)
		array[0] = 1
		cyp2.append(array.tolist())
	elif patient[37] == '*1/*2':
		array = np.zeros(12)
		array[1] = 1
		cyp2.append(array.tolist())
	elif patient[37] == '*1/*3':
		array = np.zeros(12)
		array[2] = 1
		cyp2.append(array.tolist())
	elif patient[37] == '*1/*5':
		array = np.zeros(12)
		array[3] = 1
		cyp2.append(array.tolist())
	elif patient[37] == '*1/*6':
		array = np.zeros(12)
		array[4] = 1
		cyp2.append(array.tolist())
	elif patient[37] == '*1/*11':
		array = np.zeros(12)
		array[5] = 1
		cyp2.append(array.tolist())
	elif patient[37] == '*1/*13':
		array = np.zeros(12)
		array[6] = 1
		cyp2.append(array.tolist())
	elif patient[37] == '*1/*14':
		array = np.zeros(12)
		array[7] = 1
		cyp2.append(array.tolist())
	elif patient[37] == '*2/*2':
		array = np.zeros(12)
		array[8] = 1
		cyp2.append(array.tolist())		
	elif patient[37] == '*2/*3':
		array = np.zeros(12)
		array[9] = 1
		cyp2.append(array.tolist())
	elif patient[37] == '*3/*3':
		array = np.zeros(12)
		array[10] = 1
		cyp2.append(array.tolist())
	else:
		array = np.zeros(12)
		array[11] = 1
		cyp2.append(array.tolist())
	vkorc1(vk3673, 'A/A', 'A/G', patient[41])
	vkorc1(vk5808, 'G/G', 'G/T', patient[43])
	vkorc1(vk6484, 'C/C', 'C/T', patient[45])
	vkorc1(vk6853, 'C/C', 'C/G', patient[47])
	vkorc1(vk9041, 'A/A', 'A/G', patient[49])
	vkorc1(vk7566, 'C/C', 'C/T', patient[51])
	vkorc1(vk961, 'A/A', 'A/C', patient[53])

h_max = max(height_cm)
h_min = min(height_cm)
height_cm = np.array(height_cm)
height_features = (height_cm - h_min)/(h_max - h_min)

w_max = max(weight_kg)
w_min = min(weight_kg)
weight_kg = np.array(weight_kg)
weight_features = (weight_kg - w_min)/(w_max - w_min)

# features = np.array([g + r + e + age + ind + dia + hrt + val + sm + c + v1 + v2 + v3 + v4 + v5 + v6 + v7 + b for g, r, e, age, ind, dia, hrt, val, sm, c, v1, v2, v3, v4, v5, v6, v7, b in \
# zip(gender, race, ethnicity, age_decades, diagnosis, diabetes, heart, valve, smoker, cyp2, vk3673, vk5808, vk6484, vk6853, vk9041, vk7566, vk961, bias)])

# features = np.column_stack((height_features, weight_features, features, right_dosage))

# for i in range(13, 32):
# 	print (i, set(valid_data[:, i].tolist()))

medications = np.array(medic[0])
for i in range(1, 18):
	medications = np.column_stack((medications, np.array(medic[i])))

demographics = np.array([g + r + e + age for g, r, e, age in zip(gender, race, ethnicity, age_decades)])
demographics = np.column_stack((demographics, height_features, weight_features))

diagnosis = np.array(diagnosis)

preexisting = np.array([dia + hrt + val + sm for dia, hrt, val, sm in zip(diabetes, heart, valve, smoker)])

genetics = np.array([c + v1 + v2 + v3 + v4 + v5 + v6 + v7 for c, v1, v2, v3, v4, v5, v6, v7 in \
zip(cyp2, vk3673, vk5808, vk6484, vk6853, vk9041, vk7566, vk961)])

bias = np.ones(num_patients)

features = np.column_stack((demographics, preexisting, genetics, bias, right_dosage))
# total = 0
runs = 0

# perform = np.zeros((total, runs))
# perform_avg = np.zeros(total)

# trial_ind = -1
# for trial in []:
# 	trial_ind += 1
# 	if trial == 0:
# 		features = np.column_stack((demographics, diagnosis, preexisting, medications, genetics, bias, right_dosage))
# 	elif trial == 1:
# 		features = np.column_stack((demographics, bias, right_dosage))
# 	elif trial == 2:
# 		features = np.column_stack((diagnosis, bias, right_dosage))
# 	elif trial == 3:
# 		features = np.column_stack((preexisting, bias, right_dosage))
# 	elif trial == 4:
# 		features = np.column_stack((medications, bias, right_dosage))
# 	elif trial == 5:
# 		features = np.column_stack((genetics, bias, right_dosage))
# 	elif trial == 6:
# 		features = np.column_stack((diagnosis, preexisting, medications, genetics, bias, right_dosage))
# 	elif trial == 7:
# 		features = np.column_stack((demographics, preexisting, medications, genetics, bias, right_dosage))
# 	elif trial == 8:
# 		features = np.column_stack((demographics, diagnosis, medications, genetics, bias, right_dosage))
# 	elif trial == 9:
# 		features = np.column_stack((demographics, diagnosis, preexisting, genetics, bias, right_dosage))
# 	elif trial == 10:
# 		features = np.column_stack((demographics, diagnosis, preexisting, medications, bias, right_dosage))
# 	elif trial == 11:
# 		features = np.column_stack((demographics, diagnosis, genetics, bias, right_dosage))
# 	elif trial == 12:
# 		features = np.column_stack((bias, right_dosage))
# 	elif trial == 13:
# 		features = np.column_stack((demographics, genetics, bias, right_dosage))
# 	elif trial == 14:
# 		features = np.column_stack((demographics, diagnosis, genetics, bias, right_dosage))
# 	elif trial == 15:
# 		features = np.column_stack((demographics, preexisting, genetics, bias, right_dosage))
# 	elif trial == 16:
# 		features = np.column_stack((demographics, medications, genetics, bias, right_dosage))	

	#Shuffle patients
T = num_patients

regret = np.zeros((runs, T))
avg_incorrect = np.zeros((runs, T))
pf_ucb = np.zeros((runs, T))
actions = np.zeros(T)

for n in range(runs):

	np.random.shuffle(features)
	ground_truth = features[:, -1]
	x_input = features[:, :-1]

	# LinUCB
	alpha = 0.1
	K = 3
	d = x_input.shape[1]
	A = np.eye(d).repeat(K).reshape(d, d, K)
	b = np.zeros((d, K))
	theta = np.zeros((d, K))
	p = np.zeros(K)

	#Training
	sum_incorrect = 0
	for j in range(T):
		x = x_input[j]
		for i in range(K):
			inv = np.linalg.inv(A[:, :, i])
			theta[:, i] = np.matmul(inv, b[:, i])
			bound = np.dot(x, np.matmul(inv, x))
			p[i] = np.dot(theta[:, i], x) + alpha*np.sqrt(bound)
		act = np.random.choice(np.flatnonzero(p == np.amax(p)))
		actions[j] = act
		reward = -1 if ground_truth[j] != act else 0
		regret[n, j] = regret[n, j-1] - reward if j != 0 else -reward
		sum_incorrect += -reward 
		avg_incorrect[n, j] = sum_incorrect/(j+1)
		pf_ucb[n, j] = 1 - sum_incorrect/(j+1)
		A[:, :, act] += np.outer(x, x)
		b[:, act] += reward*x


	# perform[trial_ind, n] = 1.0 - sum_incorrect/num_patients
	# print (trial, n, perform[trial_ind, n])
# perform_avg[trial_ind] = np.mean(perform[trial_ind])
# print ("Avg:", perform_avg[trial_ind])
# print (i, perform[i])
# print (np.mean(pf_linucb))

# Plots
# def plot_ci(y, y_label = 'y', x_label = 'x', title = 'Plot'):
# 	T = y.shape[1]
# 	y_mean = np.mean(y, axis = 0)
# 	y_std = np.std(y, axis = 0)
# 	z = 1.96
# 	upper_ci = y_mean + z*y_std/np.sqrt(runs)
# 	lower_ci = y_mean - z*y_std/np.sqrt(runs)

# 	CI_df = pd.DataFrame(columns = ['x_data', 'low_CI', 'upper_CI'])
# 	CI_df['x_data'] = np.arange(1, T+1)
# 	CI_df['low_CI'] = lower_ci
# 	CI_df['upper_CI'] = upper_ci
# 	CI_df.sort_values('x_data', inplace = True)

# 	_, ax = plt.subplots()
# 	# Plot the data, set the linewidth, color and transparency of the
# 	# line, provide a label for the legend
# 	ax.plot(np.arange(1, T+1), y_mean, lw = 1, color = '#539caf', alpha = 1, label = 'Mean')
# 	# Shade the confidence interval
# 	ax.fill_between(CI_df['x_data'], CI_df['low_CI'], CI_df['upper_CI'], color = '#539caf', alpha = 0.4, label = '95% CI')
# 	# Label the axes and provide a title
# 	ax.set_title(title)
# 	ax.set_xlabel(x_label)
# 	ax.set_ylabel(y_label)

# 	# Display legend
# 	ax.legend(loc = 'best')
# 	plt.show()

# plot_ci(pf_ucb, 'Performance', 'Number of Patients', 'Performance vs Number of Patients')
# plot_ci(regret, 'Regret', 'Number of Patients', 'Regret vs Number of Patients')
# plot_ci(avg_incorrect, 'Fraction of Incorrect Decisions', 'Number of Patients', 'Fraction of Incorrect Decisions vs Number of Patients')

# materials = ['All', '1+5', '1+2+5', '1+3+5', '1+4+5']
# total = len(materials)
# x_pos = np.arange(total)
# CTEs = [np.mean(perform[i]) for i in range(total)]
# error = [np.std(perform[i]) for i in range(total)]
# CTEs = [0.653183789, 0.623769898, 0.589544139, 0.614218522, 0.607706223, 0.642329956, 0.610202606]
# error = [0.006534774, 0.008584064, 0.024959103, 0.008003431, 0.001520397, 0.016777048, 0.000418809]

# CTEs = [0.653183789, 0.647539796, 0.653871201, 0.651808973, 0.658285094, 0.624131693]
# error = [0.006534774, 0.002597875, 0.004810099, 0.009068999, 0.007622626, 0.009374178]

# CTEs = [0.653183789, 0.64866465, 0.658229435, 0.66121563, 0.65402396]
# error = [0.006534774, 0.019940785, 0.005547286, 0.006167932, 0.008293749]

# print ("Avg_overall", perform_avg)
# Build the plot
# fig, ax = plt.subplots()
# ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
# ax.set_ylabel('Performance (Fraction of correct decisions)')
# ax.set_xticks(x_pos)
# ax.set_xticklabels(materials)
# ax.set_title('Choosing the right feature combination (10 permutations of patients)')
# ax.yaxis.grid(True)

# # Save the figure and show
# plt.tight_layout()
# # plt.savefig('bar_plot_with_error_bars.png')
# plt.show()


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

# np.set_printoptions(precision=2)

# # Plot non-normalized confusion matrix
# plot_confusion_matrix(ground_truth, actions, classes=np.arange(3),
#                       title='Confusion Matrix, without Normalization')

# # Plot normalized confusion matrix
# plot_confusion_matrix(ground_truth, actions, classes=np.arange(3), normalize=True,
#                       title='Normalized Confusion Matrix')

# plt.show()

# target_names = ['Arm 0 (Low)', 'Arm 1 (Medium)', 'Arm 2 (High)']
# print(classification_report(ground_truth, actions, target_names=target_names))
#LASSO Bandit

# for trial in range(1, 2):

# trial = 100

# if trial == 0:
# 	features = np.column_stack((demographics, diagnosis, preexisting, medications, genetics, bias, right_dosage))
# elif trial == 1:
# 	features = np.column_stack((demographics, bias, right_dosage))
# elif trial == 2:
# 	features = np.column_stack((diagnosis, bias, right_dosage))
# elif trial == 3:
# 	features = np.column_stack((preexisting, bias, right_dosage))
# elif trial == 4:
# 	features = np.column_stack((medications, bias, right_dosage))
# elif trial == 5:
# 	features = np.column_stack((genetics, bias, right_dosage))
# elif trial == 6:
# 	features = np.column_stack((diagnosis, preexisting, medications, genetics, bias, right_dosage))
# elif trial == 7:
# 	features = np.column_stack((demographics, preexisting, medications, genetics, bias, right_dosage))
# elif trial == 8:
# 	features = np.column_stack((demographics, diagnosis, medications, genetics, bias, right_dosage))
# elif trial == 9:
# 	features = np.column_stack((demographics, diagnosis, preexisting, genetics, bias, right_dosage))
# elif trial == 10:
# 	features = np.column_stack((demographics, diagnosis, preexisting, medications, bias, right_dosage))
# elif trial == 11:
# 	features = np.column_stack((demographics, diagnosis, genetics, bias, right_dosage))
# elif trial == 12:
# 	features = np.column_stack((demographics, diagnosis, preexisting, bias, right_dosage))
# elif trial == 13:
# 	features = np.column_stack((bias, right_dosage))
# elif trial == 14:
# 	features = np.column_stack((demographics, preexisting, genetics, bias, right_dosage))

trial = 14
features = np.column_stack((demographics, preexisting, genetics, bias, right_dosage))

runs = 0
d = features.shape[1] - 1
T = num_patients
K = 3
regret = np.zeros((runs, T))
avg_incorrect = np.zeros((runs, T))
pf_lasso = np.zeros(runs)
pf = np.zeros((runs, T))
lam1 = 0.05
q = 1
h = 5
actions = np.zeros(T)
forced = np.array([[(2**n - 1)*K*q + j for n in np.arange(11) for j in np.arange(1 + (i-1)*int(q), 1 + int(i*q))] for i in range(K)])

def estimate(X, Y, lam, x):
	clf = Lasso(alpha = lam/2,  max_iter=10000)
	clf.fit(X, Y)
	y = clf.predict(np.expand_dims(x, axis = 0))
	return y

for n in range(runs):

	np.random.shuffle(features)
	ground_truth = features[:, -1]
	x_input = features[:, :-1]
	sum_incorrect = 0
	all_samp = [[] for i in range(K)]
	rew = np.zeros(K)
	Y_t = np.zeros(T)
	lam2 = np.zeros(T+1)
	lam2[0] = 0.05

	for t in range(T):
		x = x_input[t]
		if t in forced:
			act = np.where(t == forced)[0][0]
		else:
			if t == 0:
				act = 0
			else:
				for i in range(K):
					indices = [index for index in forced[i] if index < t]
					rew[i] = estimate(x_input[indices], Y_t[indices], lam1, x)
				rew_bound = np.amax(rew) - h/2
				kappa = []
				for i in range(K):
					if rew[i] >= rew_bound:
						kappa.append(i)
				rew_max = -float('inf')
				for k in kappa:
					indices = [index for index in all_samp[k] if index < t]
					reward_all = estimate(x_input[indices], Y_t[indices], lam2[t], x)
					if reward_all >= rew_max:
						rew_max = reward_all
						act = k
		all_samp[act].append(t)
		lam2[t+1] = lam2[0]*np.sqrt(np.log((t+1)*d)/(t+1))
		reward = -1 if ground_truth[t] != act else 0
		Y_t[t] = reward
		regret[n, t] = regret[n, t-1] - reward if t != 0 else -reward
		sum_incorrect += -reward 
		pf[n, t] = 1 - sum_incorrect/(t+1)
		print (n, t, pf[n, t])
		avg_incorrect[n, t] = sum_incorrect/(t+1.0)
		actions[t] = act

	pf_lasso[n] = 1.0 - sum_incorrect/T
	print (trial, n, "Performance", pf_lasso)

# np.set_printoptions(precision=2)

# # Plot non-normalized confusion matrix
# plot_confusion_matrix(ground_truth, actions, classes=np.arange(3),
#                       title='Confusion Matrix, without Normalization')

# # Plot normalized confusion matrix
# plot_confusion_matrix(ground_truth, actions, classes=np.arange(3), normalize=True,
#                       title='Normalized Confusion Matrix')

# plt.show()

# target_names = ['Arm 0 (Low)', 'Arm 1 (Medium)', 'Arm 2 (High)']
# print(classification_report(ground_truth, actions, target_names=target_names))

# print (trial, pf_lasso)
# plot_ci(pf, 'Performance', 'Number of Patients', 'Performance vs Number of Patients')
# plot_ci(regret, 'Regret', 'Number of Patients', 'Regret vs Number of Patients')
# plot_ci(avg_incorrect, 'Fraction of Incorrect Decisions', 'Number of Patients', 'Fraction of Incorrect Decisions vs Number of Patients')

