"""
File: titanic_level2.py
Name: 
----------------------------------
This file builds a machine learning algorithm by pandas and sklearn libraries.
We'll be using pandas to read in dataset, store data into a DataFrame,
standardize the data by sklearn, and finally train the model and
test it on kaggle website. Hyper-parameters tuning are not required due to its
high level of abstraction, which makes it easier to use but less flexible.
You should find a good model that surpasses 77% test accuracy on kaggle.
"""

import math
import pandas as pd
from sklearn import preprocessing, linear_model

TRAIN_FILE = 'titanic_data/train.csv'
TEST_FILE = 'titanic_data/test.csv'


def data_preprocess(filename, mode='Train', training_data=None):
	"""
	:param filename: str, the filename to be read into pandas
	:param mode: str, indicating the mode we are using (either Train or Test)
	:param training_data: DataFrame, a 2D data structure that looks like an excel worksheet
						  (You will only use this when mode == 'Test')
	:return: Tuple(data, labels), if the mode is 'Train'; or return data, if the mode is 'Test'
	"""
	data = pd.read_csv(filename)
	labels = None

	data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)  # Delete these columns

	data.loc[data.Sex == 'male', 'Sex'] = 1
	data.loc[data.Sex == 'female', 'Sex'] = 0

	data.loc[data.Embarked == 'S', 'Embarked'] = 0
	data.loc[data.Embarked == 'C', 'Embarked'] = 1
	data.loc[data.Embarked == 'Q', 'Embarked'] = 2

	if mode == 'Train':
		data.dropna(inplace=True)  # Delete rows with NaN values
		labels = data.pop('Survived')
	else:
		# mode == 'Test'
		training_data_means = {col: round(training_data[col].mean(), 3) for col in ['Age', 'Fare']}
		for col in ['Age', 'Fare']:
			data[col].fillna(training_data_means[col], inplace=True)

	if mode == 'Train':
		return data, labels
	elif mode == 'Test':
		return data


def one_hot_encoding(data, feature):
	"""
	:param data: DataFrame, key is the column name, value is its data
	:param feature: str, the column name of interest
	:return data: DataFrame, remove the feature column and add its one-hot encoding features
	"""
	data = pd.get_dummies(data, columns=[feature])
	if feature == 'Pclass':
		data.rename(columns={'Pclass_1': 'Pclass_0', 'Pclass_2': 'Pclass_1', 'Pclass_3': 'Pclass_2'}, inplace=True)

	return data


def standardization(data, mode='Train'):
	"""
	:param data: DataFrame, key is the column name, value is its data
	:param mode: str, indicating the mode we are using (either Train or Test)
	:return data: DataFrame, standardized features
	"""
	if mode == 'Train':
		std_scaler = preprocessing.StandardScaler()
		data = std_scaler.fit_transform(data)

	return data


def main():
	"""
	You should call data_preprocess(), one_hot_encoding(), and
	standardization() on your training data. You should see ~80% accuracy on degree1;
	~83% on degree2; ~87% on degree3.
	Please write down the accuracy for degree1, 2, and 3 respectively below
	(rounding accuracies to 8 decimal places)
	real accuracy on degree1 -> 0.80196629
	real accuracy on degree2 -> 0.83707865
	real accuracy on degree3 -> 0.87640449
	"""
	train_data = data_preprocess(TRAIN_FILE, 'Train')
	train_x = train_data[0]
	y = train_data[1]

	train_x = one_hot_encoding(train_x, 'Sex')
	train_x = one_hot_encoding(train_x, 'Pclass')
	train_x = one_hot_encoding(train_x, 'Embarked')

	std_scaler = preprocessing.StandardScaler()
	train_x = std_scaler.fit_transform(train_x)

	poly_d2 = preprocessing.PolynomialFeatures(degree=2)
	train_x_2 = poly_d2.fit_transform(train_x)

	poly_d3 = preprocessing.PolynomialFeatures(degree=3)
	train_x_3 = poly_d3.fit_transform(train_x)

	classifier = linear_model.LogisticRegression(max_iter=1000)

	print('degree1: ', classifier.fit(train_x, y).score(train_x, y))
	print('degree2: ', classifier.fit(train_x_2, y).score(train_x_2, y))
	print('degree3: ', classifier.fit(train_x_3, y).score(train_x_3, y))


if __name__ == '__main__':
	main()
