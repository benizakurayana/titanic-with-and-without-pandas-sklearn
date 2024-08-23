"""
File: titanic_level1.py
Name: Jane
----------------------------------
This file builds a machine learning algorithm from scratch 
by Python. We'll be using 'with open' to read in dataset,
store data into a Python dict, and finally train the model and 
test it on kaggle website. This model is the most flexible among all
levels. You should do hyper-parameter tuning to find the best model.
"""

import math
import util
TRAIN_FILE = 'titanic_data/train.csv'
TEST_FILE = 'titanic_data/test.csv'


def data_preprocess(filename: str, data: dict, mode='Train', training_data=None):
	"""
	:param filename: str, the filename to be processed
	:param data: an empty Python dictionary
	:param mode: str, indicating if it is training mode or testing mode
	:param training_data: dict[str: list], key is the column name, value is its data
						  (You will only use this when mode == 'Test')
	:return data: dict[str: list], key is the column name, value is its data
	"""
	data = {}

	with open(filename, 'r') as f:
		is_head = True
		for line in f:
			cols = line.strip().split(',')
			if is_head:
				for col in cols:
					data[col] = []
				data.pop('PassengerId')
				data.pop('Name')
				data.pop('Ticket')
				data.pop('Cabin')
				is_head = False

			elif mode == 'Train':
				has_missing_data = False
				col_indices = [1, 2, 5, 6, 7, 8, 10, 12]  # Wanted columns
				for i in col_indices:
					if cols[i] == '':
						has_missing_data = True
						break
				if not has_missing_data:  # Only process the rows without missing data in wanted columns
					for i in col_indices:
						if i == 1:
							data['Survived'].append(int(cols[i]))
						elif i == 2:
							data['Pclass'].append(int(cols[i]))
						elif i == 5:
							data['Sex'].append(1 if cols[i] == 'male' else 0)
						elif i == 6:
							data['Age'].append(float(cols[i]))
						elif i == 7:
							data['SibSp'].append(int(cols[i]))
						elif i == 8:
							data['Parch'].append(int(cols[i]))
						elif i == 10:
							data['Fare'].append(float(cols[i]))
						elif i == 12:
							if cols[i] == 'S':
								data['Embarked'].append(0)
							elif cols[i] == 'C':
								data['Embarked'].append(1)
							elif cols[i] == 'Q':
								data['Embarked'].append(2)

			else:
				# mode == 'Test'
				training_data_means = {'Age': round(sum(training_data['Age']) / len(training_data['Age']), 3),
									   'Fare': round(sum(training_data['Fare']) / len(training_data['Fare']), 3)}
				col_indices = [1, 4, 5, 6, 7, 9, 11]  # Wanted columns
				for i in col_indices:
					if i == 1:
						data['Pclass'].append(int(cols[i]))
					elif i == 4:
						data['Sex'].append(1 if cols[i] == 'male' else 0)
					elif i == 5:
						if cols[i] == '':
							cols[i] = training_data_means['Age']
						data['Age'].append(float(cols[i]))
					elif i == 6:
						data['SibSp'].append(int(cols[i]))
					elif i == 7:
						data['Parch'].append(int(cols[i]))
					elif i == 9:
						if cols[i] == '':
							cols[i] = training_data_means['Fare']
						data['Fare'].append(float(cols[i]))
					elif i == 11:
						if cols[i] == 'S':
							data['Embarked'].append(0)
						elif cols[i] == 'C':
							data['Embarked'].append(1)
						elif cols[i] == 'Q':
							data['Embarked'].append(2)

	return data


def one_hot_encoding(data: dict, feature: str):
	"""
	:param data: dict[str, list], key is the column name, value is its data
	:param feature: str, the column name of interest
	:return data: dict[str, list], remove the feature column and add its one-hot encoding features
	"""
	# Add one-hot encoding feature columns
	unique_values = list(set(data[feature]))  # set(data[feature]) creates a set by finding the unique elements of the list and sorting them in ascending order
	for i in range(len(unique_values)):
		data[feature + '_' + str(i)] = [1 if val == unique_values[i] else 0 for val in data[feature]]

	# Remove the feature column
	data.pop(feature)

	return data


def normalize(data: dict):
	"""
	:param data: dict[str, list], key is the column name, value is its data
	:return data: dict[str, list], key is the column name, value is its normalized data
	"""
	# X_normalized = (X - X_min) / (X_max - X_min)
	for col in data:
		X_min = min(data[col])
		X_max = max(data[col])
		X_maxmin = X_max - X_min
		for i in range(len(data[col])):
			data[col][i] = (data[col][i] - X_min) / X_maxmin

	return data


def learnPredictor(inputs: dict, labels: list, degree: int, num_epochs: int, alpha: float):
	"""
	:param inputs: dict[str, list], key is the column name, value is its data
	:param labels: list[int], indicating the true label for each data
	:param degree: int, degree of polynomial features
	:param num_epochs: int, the number of epochs for training
	:param alpha: float, known as step size or learning rate
	:return weights: dict[str, float], feature name and its weight
	"""
	# Step 1 : Initialize weights
	weights = {}  # feature => weight
	keys = list(inputs.keys())
	if degree == 1:
		for i in range(len(keys)):
			weights[keys[i]] = 0
	elif degree == 2:
		for i in range(len(keys)):
			weights[keys[i]] = 0
		for i in range(len(keys)):
			for j in range(i, len(keys)):
				weights[keys[i] + keys[j]] = 0

	# Step 2 : Start training
	for epoch in range(num_epochs):
		def sigmoid(k):
			return 1 / (1 + math.exp(-k))

		for i in range(len(inputs[keys[0]])):
	# Step 3 : Feature Extract
			y = labels[i]
			phi_x = {key: inputs[key][i] for key in keys}
			if degree == 2:
				for m in range(len(keys)):
					for n in range(m, len(keys)):
						phi_x[keys[m] + keys[n]] = inputs[keys[m]][i] * inputs[keys[n]][i]

	# Step 4 : Update weights
			# w = w - alpha*((h-y)phi_x)
			h = sigmoid(util.dotProduct(weights, phi_x))
			scale = -alpha*(h-y)
			util.increment(weights, scale, phi_x)

	return weights
