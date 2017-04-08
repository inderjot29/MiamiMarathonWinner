import csv
import NormalizeData as nd
import numpy as np
from numpy import genfromtxt
import pandas  as pd
import warnings
import preprocess
from sklearn import metrics

import matplotlib.pyplot as plt
from sklearn import cross_validation
import math
warnings.filterwarnings("ignore")

#global vairables
probability_Y0=0
probability_Y1=0
totalY=0
Y1_value=0
Y0_value=0
featureCategories={}
summarizedMeanStd={}
X_instances_per_class_1=[]
X_instances_per_class_0=[]
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
	
def ApplyNaiveBayes(X):
	global summarizedMeanStd
	Y= genfromtxt('Y1.csv',skip_header=1,delimiter=',')
	NaiveBayesGaussian(X,Y)
	"""#choose second column
	Y1=Y[:, [1]]
	
	#choose the columns age,gender,runningtime average, pace
	new_dataFrame=X[["Age","Sex","Year"]]"""
	"""plt.plot(X[['Year']],X[['Sex']],'ro')
	plt.xlabel('Year')
	plt.ylabel('Time')
	plt.show()"""
	
	
def NaiveBayesGaussian(X,Y):
	#print Y.shape
	index = [i for i in range(1, len(Y)+1)]
	dfY = pd.DataFrame(Y,index=index,columns=['Id','Participation'])
	GetProbabilityOfY(dfY)
	#choose second column
	Y1=Y[:, [1]]
	
	#here implement the algortihm
	#choose the columns age,gender,runningtime average, pace
	new_dataFrame=X[["Age","Sex","Year"]]
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(new_dataFrame, Y1, test_size=0.4, random_state=0)
	#summarizedMeanStd=CreateXPerClass(X_train,y_train)
	summarizedMeanStd=CreateXPerClass(X_train,y_test)
	predictions=getPredictions(summarizedMeanStd,X_train.values)
	print predictions[0:20]
	print Y1[40:60,[0]]
	print getAccuracy(y_train,predictions)
	predict2017Values()
	
def calculateProbability(x, mean, stdev):
	#for i in xrange( len( x ) ):
		#x[i,:] = x[i,:] - mean
	exponent = np.exp(-(np.power(x-mean,2)/float(2*np.power(stdev,2))))
	return (1 / (np.sqrt(2*np.pi) * stdev)) * exponent
 
def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.iteritems():
		probabilities[classValue]=1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue]= probabilities[classValue]*calculateProbability(x, mean, stdev)
	return probabilities
			
	
def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	"""bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.iteritems():
		if bestLabel is None or probability > bestProb:
			if classValue==0:
				bestProb = probability*probability_Y0
				bestLabel = classValue
			else:
				bestProb = probability*probability_Y1
				bestLabel = classValue"""
	if bool(probabilities)!=False:
		Y0=probabilities[0]*probability_Y0
		Y1=probabilities[1]*probability_Y1
		final_probability_1=Y1/float(Y1+Y0)
		final_probability_0=Y0/float(Y1+Y0)
		if final_probability_1>= final_probability_0:
			return 1
		else:
			return 0

 
def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[[i],:][0])
		predictions.append(result)
	return predictions
 
def getAccuracy(testSet, predictions):
	correct = 0
	ones=0
	zeros=0
	for i in range(len(testSet)):
		if testSet[i][0]== predictions[i]:
			correct += 1
		if predictions[i]==1:
			ones=ones+1
		elif predictions[i]==0:
			zeros=zeros+1
	print ones,zeros
	return (float(correct)/float(len(predictions))) * 100.0

def calculateMeanAndStd(X):
	list_age=[]
	list_sex=[]
	list_year=[]
	X.values[0].std(ddof=1)
	mean_age=X.values[0].mean()
	std_age=X.values[0].std(ddof=1)
	list_age.append(mean_age)
	list_age.append(std_age)
	mean_sex=X.values[1].mean()
	std_sex=X.values[1].std(ddof=1)
	list_sex.append(mean_sex)
	list_sex.append(std_sex)
	mean_year=X.values[2].mean()
	std_year=X.values[2].std(ddof=1)
	list_year.append(mean_year)
	list_year.append(std_year)
	return tuple(list_age),tuple(list_sex),tuple(list_year)
	
def CreateXPerClass(X,Y1):
	global X_instances_per_class_1,X_instances_per_class_0
	for instance,classLabel in zip(X.values,Y1):
		if classLabel==1:
			X_instances_per_class_1.append(tuple(instance))
		elif classLabel==0:
			X_instances_per_class_0.append(tuple(instance))
	X_instances_per_class_0=pd.DataFrame.from_records(X_instances_per_class_0)
	#print X_instances_per_class_0.describe()
	X_instances_per_class_1=pd.DataFrame.from_records(X_instances_per_class_1)
	#print X_instances_per_class_1.describe()
	age_0,sex_0,year_0=calculateMeanAndStd(X_instances_per_class_0)
	age_1,sex_1,year_1=calculateMeanAndStd(X_instances_per_class_1)
	list_of_vectors_1=[]
	list_of_vectors_1.append(age_1)
	list_of_vectors_1.append(sex_1)
	list_of_vectors_1.append(year_1)
	
	list_of_vectors_0=[]
	list_of_vectors_0.append(age_0)
	list_of_vectors_0.append(sex_0)
	list_of_vectors_0.append(year_0)
	
	meanAndStdPerclassLabel={}
	meanAndStdPerclassLabel[1]=list_of_vectors_1
	meanAndStdPerclassLabel[0]=list_of_vectors_0
	return meanAndStdPerclassLabel
			

	
def GetProbabilityOfY(Y):
	global probability_Y0,probability_Y1,totalY,Y1_value,Y0_value
	totalY=Y['Participation'].count()
	Y1_value=Y['Participation'].astype(bool).sum()
	Y0_value=totalY-Y1_value
	probability_Y1=Y1_value/float(totalY)
	probability_Y0=1-probability_Y1
	

	


def predict2017Values():
	X=getPandasDataFrame('Project1_data_edit3.csv')
	
	# X is the vector which we give as an input to our algorithm
	new_dataFrame=X[["Age","Sex","Year"]]
	#print new_dataFrame.head()
	X1=new_dataFrame.values
	
	predict_2017=getPredictions(summarizedMeanStd,X1)
	
	predict_2017=np.reshape(predict_2017, (-1, 1))
	print predict_2017.shape
	print X['Id'].shape
	runtime_with_id=np.concatenate((X['Id'],predict_2017),axis=1)
	with open('2017_LogisticRegression.csv', 'wb') as out:
		writer = csv.writer(out)
		fieldnames = [('Id', 'Participation')]
		writer.writerow(fieldnames[0])
		for row in runtime_with_id:
			writer.writerow(row)
	
	
	
def getPandasDataFrame(fileName):
	preprocess.PreProcessing(fileName)
	orignial_data= genfromtxt('output.csv', delimiter=',',usecols=(0,2,3,4,5,6,7))
	#it normalizes rank, pace and running time only. Returns the normalized array. Also it normalizes categorical data.
	normalized_data=nd.normalizeContinuousData(orignial_data)
	#print normalized_data[0]
	newFeatureData=nd.addingNewFeatureYearRangeAndCount(normalized_data)
	#print newFeatureData[0]
	final_data=nd.replaceAgeWithRangeLabel(newFeatureData)
	#print final_data.shape
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
	#print df.shape
	#print final_dataframe.head()
	# X is the vector which we give as an input to our algorithm
	
	ApplyNaiveBayes(df)
