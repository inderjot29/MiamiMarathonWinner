"""
Created on Fri Jan 19,2017
@author: Inderjot Kaur Ratol
"""

import csv
import numpy as np
import sklearn.preprocessing as pp
from sklearn.preprocessing import normalize
from numpy import genfromtxt


#global variables
currentId=0
yearCount=0
id_yearCount = {}
id_yearRange={}
id_AgeRange={}
currentYear=0
previousYear=0
currentAge=0
yearRangeCategory={}
ageRangeMax=[19,24,29,34,39,44,49,54,59,64,69,74,79]
ageRangeMin=[15,20,25,30,35,40,45,50,55,60, 65,70,75]
yearRange=[2016,2015,2014,2013,2012,2011,2010,2009,2008,2007,2006,2005,2004,2003]
yearRangeLabel=[5,5,4,4,4,3,3,3,2,2,2,1,1,1]

""" Initialze the current id global variable 
to keep a track of the ids while modfiying the excel"""
def InitializeCurrentId(id):
	global currentId
	if(currentId==0):
		currentId=int(id)
	elif(id!=currentId):
		global id_yearCount
		global yearCount
		id_yearCount[currentId]=yearCount
		currentId=int(id)
		yearCount=0

""" Normalize the data in column rank, time and pace """
def normalizeContinuousData(orignial_data):
	#normalizd rank,running time and pace
	normalized_data = normalize(orignial_data[:, [3,4,5]], axis=0) #axis=0 for column-wise
	year_data=orignial_data[:, [6]]
	#normalize categorical data
	#gender_data=normalizeCategoricalData(orignial_data)
	#id_age=np.concatenate(( orignial_data[:,[0,1,2]], gender_data), axis=1)
	normalized_data=np.concatenate(( orignial_data[:,[0,1,2]], normalized_data), axis=1)
	normalized_data=np.concatenate((normalized_data,year_data), axis=1)
	return normalized_data
	
""" Normalize the categorical data, i.e. gender and age after changing age into range values """
def normalizeCategoricalData(data):
	label_encoder = pp.LabelEncoder()
	#enc = pp.OneHotEncoder(n_values=[3],dtype='str')
	#gender_data=orignial_data[:, [3]]
	#normalized_data=enc.fit(gender_data)
	# Convert Sex variable to numeric
	normalized_data=label_encoder.fit_transform(data)
	normalized_data=np.reshape(normalized_data, (-1, 1))
	
	return normalized_data

""" Adds year count asa new feature , which is then multiplied with the normalized year value to obtaind the new feature."""
def addYearCount(year_data):
	global currentId,yearCount,id_yearCount
	currentId=0
	yearCount=0
	id_yearCount={}
	for row in year_data:
		InitializeCurrentId(row[0])
		if row[0]==currentId:
			yearCount=yearCount+1
	#adding the last row data	
	id_yearCount[currentId]=yearCount
	
	for row in year_data:
		row[1]=id_yearCount[row[0]]
	
	return year_data[:, [1]]

""" Replacing the year value by range and 
inserting labels for the range value using yearRangeCategory dictionary created with labels and range"""
def replaceYearByRange(year_data):
	global currentId,currentYear,id_yearRange,yearRangeCategory
	currentId=0
	currentYear=0
	id_yearRange={}
	for row in year_data:
		
		InitializeCurrentId(row[0])
		if row[0]==currentId:
			currentYear=int(row[1])
			
			if currentId in id_yearRange: 
				if yearRangeCategory[currentYear]> id_yearRange[currentId]:
					id_yearRange[currentId]=yearRangeCategory[currentYear]
			else:
				yearLabel=yearRangeCategory[currentYear]
				id_yearRange[currentId]=yearLabel
	
	for row in year_data:
		row[1]=id_yearRange[row[0]]
	
	return year_data[:, [1]]
	
def createYearRangeDictionary():
	global yearRangeCategory,yearRangeLabel,yearRange
	for i in range(len(yearRange)):
		yearRangeCategory[yearRange[i]] = yearRangeLabel[i]

"""adding new feature by multiplying the number of years participated and range label of years"""
def addingNewFeatureYearRangeAndCount(orignial_data):
	year_data=orignial_data[:, [0,6]]
	#replacing year data with count and range label by adding new feature.
	year_count=addYearCount(year_data)
	"""create dictionary with labels for year and range 
	where yearRange list contains the maximum value of the range.
	All the ranges used are mutually exclusive."""
	year_data=orignial_data[:, [0,6]]
	createYearRangeDictionary()
	year_column_replaced=replaceYearByRange(year_data)
	year_final_data=np.multiply(year_count,year_column_replaced)
	""" Do not normailize for the naive bayes"""
	year_final_data = normalize(year_final_data, axis=0)
	normalized_data=np.concatenate((orignial_data[:, [0,1,2,3,4,5]],year_final_data), axis=1)

	return normalized_data

"""Replacing age with range labels"""
def replaceAgeWithRangeLabel(orignial_data):
	age_data=orignial_data[:, [0,1]]
	#replacing age data age group label.
	global currentId,currentAge
	currentId=0
	currentAge=0
	#print age_data[0]
	for row in age_data:
		InitializeCurrentId(row[0])
		if row[0]==currentId:
			currentAge=int(row[1])
			if currentId in id_AgeRange: 
				if id_yearRange[currentId]<currentAge:
					for i in range(len(ageRangeMax)):
						if currentAge<= ageRangeMax[i] and currentAge>=ageRangeMin[i]:
							id_AgeRange[currentId]=(ageRangeMax[i]+ageRangeMin[i])/2
			else:
				for i in range(len(ageRangeMax)):
					if currentAge<= ageRangeMax[i] and currentAge>=ageRangeMin[i]:
						id_AgeRange[currentId]=(ageRangeMax[i]+ageRangeMin[i])/2
				if currentAge>79:
					id_AgeRange[currentId]=80
	
	for row in age_data:
		row[1]=id_AgeRange[row[0]]
	""" Left for the Naive Bayes part"""
	age_data_labelled=normalize(age_data[:, [1]],axis=0)
	normalized_data=np.concatenate((orignial_data[:, [0]] ,age_data_labelled ), axis=1)
	normalized_data=np.concatenate((normalized_data , orignial_data[:, [2,3,4,5,6]]), axis=1)
	return normalized_data

	