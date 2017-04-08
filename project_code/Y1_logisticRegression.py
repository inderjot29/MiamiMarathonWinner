
import csv
import NormalizeData as nd
import numpy as np
from numpy import genfromtxt
import pandas  as pd
import warnings
import preprocess
import matplotlib.pyplot as plt
from sklearn import cross_validation
import math

def GetFinalInputData(df):
	aggregations = {
	'Id': { # work on the "age" column
	'id': 'max' # get the max, and call this result 'age',(just to get all data in one go, otheriwse we have already set the mean value)
		},
	'Age': { # work on the "age" column
	'age': 'max' # get the max, and call this result 'age',(just to get all data in one go, otheriwse we have already set the mean value)
		},
	'Sex': { # work on the "sex" column
	'sex': 'max' # get the max, and call this result 'sex',same explanation as above
		},
	'Rank': { # work on the "rank" column
	'rank': 'mean' # get the mean, and call this result 'rank'
		},
	'Time': { # work on the "time" column
	'time': 'mean' # get the mean, and call this result 'time'
		},
	'Pace': { # work on the "pace" column
	'pace': 'mean' # get the mean, and call this result 'pace'
		},
	'Year': { # work on the "year" column
	'year': 'max' # get the max, and call this result 'year'
		},
	}
	final_vector=df.groupby('Id').agg(aggregations)
	#print final_vector['Age']
	return final_vector 
 
# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	print len(actual)-correct
	return correct / float(len(actual)) * 100.0
 
# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(X,Y, alpha,epoch):
	scores = list()
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.4, random_state=0)
	predicted = logistic_regression(X, Y, alpha,epoch)
	print predicted[0:20]
	accuracy = accuracy_metric(Y, predicted)
	scores.append(accuracy)
	return scores
 
# Make a prediction with coefficients
"""def predict(row, coefficients):
	JTheta = 0
	for i in range(len(row)-1):
		JTheta += coefficients[i] * row[i]
	return sigmoid(JTheta)"""

def predict(X,theta):
	print theta,"theta"
	probability = sigmoid(X * theta.T)
	print probability[0:5]
	return [1 if x >= 0.5 else 0 for x in probability]

def sigmoid(z):  
    return 1 / (1 + np.exp(-z))
 
# Estimate logistic regression coefficients using stochastic gradient descent
def coefficients_sgd(train,test, l_rate, iterations):
	#Init Theta and Run Gradient Descent
	J_history = np.zeros(shape=(iterations, 1))
	coef = np.zeros(train.shape[1])
	coef = np.matrix(coef)
	print coef.shape
	for i in range(iterations):
		print "iteraion",i
		for row in train:
			JTheta=sigmoid(row*coef.T)
			error = JTheta-test[i]
			for j in range(len(row)-1):
				if j==0:
					coef[:,[0]] = coef[:,[0]] + l_rate * error * JTheta * (1.0 - JTheta)
				else:
					coef[:,[j]] = coef[:,[j]] + l_rate * error * JTheta * (1.0 - JTheta) * row[j]
		J_history[i, 0] = compute_cost(train, test, coef)
	return coef,J_history

def compute_cost(X, y, theta):
	theta = np.matrix(theta)
	X = np.matrix(X)
	y = np.matrix(y)
	first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
	second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
	return np.sum(first - second) / (len(X))

 
# Linear Regression Algorithm With Stochastic Gradient Descent
def logistic_regression(train, test, alpha, iterations):
	predictions = list()
	coef,J_history = coefficients_sgd(train,test, alpha, iterations)
	print coef
	predictions = predict(train, coef)
	plt.figure(1)
	plt.plot(np.arange(iterations), J_history)
	plt.xlabel('Iterations')
	plt.ylabel('Cost Function')
	plt.show()
	return(predictions)
	
def ApplyLogisticRegression(X):
	# evaluate algorithm

	alpha = 0.00001
	iterations = 50
	Y= genfromtxt('Y1.csv',skip_header=1,delimiter=',')
	
	#choose second column
	Y1=Y[:, [1]]
	final_Y=Y1.ravel()
	#choose the columns age,gender,runningtime average, pace
	X1=X[["Age","Sex","Year","YearCount","YearCategory"]]
	

	m = float(len(X1))
	#Add a column of ones to X (interception data)
	biasterms = np.ones(shape=(m, 1))
	final_X1=np.concatenate((biasterms,X1),axis=1)
	scores = evaluate_algorithm(final_X1, Y1, alpha, iterations)
	print('Scores: %s' % scores)
	print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
 
# Test the logistic regression algorithm on the diabetes dataset
def getPandasDataFrame(fileName):
	preprocess.PreProcessing(fileName)
	orignial_data= genfromtxt('output.csv', delimiter=',',usecols=(0,2,3,4,5,6,7))
	#it normalizes rank, pace and running time only. Returns the normalized array. Also it normalizes categorical data.
	normalized_data=nd.normalizeContinuousData(orignial_data)
	#print normalized_data[0]
	newFeatureData=nd.addingNewFeatureYearRangeAndCount(normalized_data)
	#print newFeatureData[0]
	final_data=nd.replaceAgeWithRangeLabel(newFeatureData)
	print final_data.shape
	#column_Names=np.array(['Id','Age','Sex','Rank','Time','Pace','Year'])
	index = [i for i in range(1, len(final_data)+1)]
	#data_with_columns=np.concatenate((column_Names ,final_data ), axis=0)
	df = pd.DataFrame(final_data,index=index,columns=['Id','Age','Sex','Rank','Time','Pace','Year'])
	final_dataframe=GetFinalInputData(df)
	return final_dataframe


if __name__ == "__main__":
	""" it assumes that CSV calues are delimited using commas,
	any spurious data column containing commas should be corrected before hand"""
	print "Enter file name with absolute path and extension :"
	fileName=raw_input()
	df=getPandasDataFrame(fileName)
	#print final_dataframe.head()
	ApplyLogisticRegression(df)

