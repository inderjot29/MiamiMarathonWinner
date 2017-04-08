import csv
import NormalizeData as nd
import numpy as np
from numpy import genfromtxt
import pandas  as pd
import warnings
import preprocess
from sklearn import metrics
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn import cross_validation
import os

# this allows plots to appear directly in the notebook

warnings.filterwarnings("ignore")


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
	

	
def ApplyLinearRegression(df):
	Y= genfromtxt('Y2.csv',skip_header=1,delimiter=',')
	#choose second column
	Y2=Y[:, [1]]
	#choose the columns age,gender,runningtime average, pace
	new_dataFrame=df[["Age","Sex","Pace"]]
	#print new_dataFrame.head()
	X2=new_dataFrame.values
	GradientDescentMethod(X2,Y2)
	
def gradient_descent(X, y, theta, alpha, num_iters):
	'''
	Performs gradient descent to learn theta
	by taking num_items gradient steps with learning
	rate alpha
	'''
	m = y.size
	print X.shape
	J_history = np.zeros(shape=(num_iters, 1))

	for i in range(num_iters):

		predictions = X.dot(theta)

		theta_size = theta.size

		for it in range(theta_size):

			temp = X[:, it]
			temp.shape = (m, 1)

			errors_x1 = (predictions - y) * temp

			theta[it][0] = theta[it][0] - alpha * (1.0 / m) * errors_x1.sum()

		J_history[i, 0] = compute_cost(X, y, theta)

	return theta, J_history
	
def feature_normalize(X):
	'''
	Returns a normalized version of X where
	the mean value of each feature is 0 and the standard deviation
	is 1. This is often a good preprocessing step to do when
	working with learning algorithms.
	'''
	mean_r = []
	std_r = []

	X_norm = X

	n_c = X.shape[1]
	for i in range(n_c):
		m = np.mean(X[:, i])
		s = np.std(X[:, i])
		mean_r.append(m)
		std_r.append(s)
		X_norm[:, i] = (X_norm[:, i] - m) / s

	return X_norm, mean_r, std_r


def compute_cost(X, y, theta):
	'''
	Comput cost for linear regression
	'''
	#Number of training samples
	m = y.size
	predictions = X.dot(theta)

	sqErrors = (predictions - y)

	J = (1.0 / (2 * m)) * sqErrors.T.dot(sqErrors)

	return J

def GradientDescentMethod(X2,Y2):
	#Scale features and set them to zero mean
	X2, mean_r, std_r = feature_normalize(X2)
	print X2.shape,Y2.shape
	m = float(len(X2))
	#Add a column of ones to X (interception data)
	biasterms = np.ones(shape=(m, 1))
	final_X2=np.concatenate((biasterms,X2),axis=1)
	
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(final_X2, Y2, test_size=0.4, random_state=0)
	print X_train.shape
	#Some gradient descent settings
	iterations = 10000
	alpha = 0.001

	#Init Theta and Run Gradient Descent
	theta = np.zeros(shape=(final_X2.shape[1], 1))

	theta, J_history = gradient_descent(X_train, y_train, theta, alpha, iterations)
	print theta[0:5],J_history[0:5]
	
	predicted_value=final_X2.dot(theta)
	print metrics.f1_score(predicted_value,Y2)
	print predicted_value[0:5]
	
	#predict run time for 2017. Load file and predict using the theta obtained
	#predict2017Values(theta)
	
	
	theta_test, J_history_test = gradient_descent(X_test, y_test, theta, alpha, iterations)
	plt.figure(1)

	#plt.plot(np.arange(iterations), J_history,'r'),np.arange(iterations), J_history_test,'b')
	train_error=plt.plot(np.arange(iterations), J_history,'r')
	test_error=plt.plot(np.arange(iterations), J_history_test,'b')
	plt.legend([train_error, test_error], ['Training Error', 'Testing Error'])
	plt.xlabel('Iterations')
	plt.ylabel('Cost Function')
	plt.show()
	
def predict2017Values(theta):
	X=getPandasDataFrame('Project1_data_edit3.csv')
	# X is the vector which we give as an input to our algorithm
	new_dataFrame=X[["Age","Sex","Pace"]]
	#print new_dataFrame.head()
	X2=new_dataFrame.values
	X2, mean_r, std_r = feature_normalize(X2)
	m = float(len(X2))
	#Add a column of ones to X (interception data)
	biasterms = np.ones(shape=(m, 1))
	final_X2=np.concatenate((biasterms,X2),axis=1)
	predict_2017_time=final_X2.dot(theta)
	runtime_with_id=np.concatenate((X['Id'],predict_2017_time),axis=1)
	with open('2017_runtime.csv', 'wb') as out:
		writer = csv.writer(out)
		fieldnames = [('Id', 'Time')]
		writer.writerow(fieldnames[0])
		for row in runtime_with_id:
			writer.writerow(row)
	#os.remove('output.csv')

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
	ApplyLinearRegression(df)