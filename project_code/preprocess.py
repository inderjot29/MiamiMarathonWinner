
"""
Created on Fri Jan 19,2017
@author: Inderjot Kaur Ratol
"""
import csv
import subprocess
import numpy as np
import sklearn.preprocessing as pp
import os


#global variables
currentId=0
previousage=0
previousyear=0
idCount=0
last_id=30417
yearsPerId=[]

""" Removes the records which has 0 value as age"""
def RemoveZeroAge(row,writer):
	age=int(row[2])
	time=row[5]
	hr, min, sec = [float(x) for x in time.split(':')]
	if age != 0 and age>14 :
			writer.writerow(row)
	
		
	
""" Initialze the current id global variable 
to keep a track of the ids while modfiying the excel"""
def InitializeCurrentId(id):
	global currentId,yearsPerId
	if(currentId==0):
		currentId=id
		yearsPerId=[]
	elif(id!=currentId):
		currentId=id
		global previousage
		previousage=0
		global previousyear
		previousyear=0
		global idCount
		idCount=0
		yearsPerId=[]

""" Initialze the previous year global variable 
to keep track of the year which has already been processed"""
def InitializeYear(year):
	global yearsPerId
	#if(previousyear==0):
	yearsPerId.append(year)

""" Increment the age of the runner when previous
 year's age is greater than the current year"""
def IncrementAge(row,writer):
	id=int(row[0])
	InitializeCurrentId(id)
	if(id==currentId):
		age=int(row[2])
		year=int(row[7])
		#initialize previous age and previous year
		global previousage
		if(previousage==0):
			previousage=age
		global previousyear
		if(previousyear==0):
			previousyear=year
		"""check if the previous age is equal or greater than the current age,
		increment by the diference between the years"""
		if(age<=previousage):
			age=previousage+(year-previousyear)
			row[2]=age
			previousage=age
		else:
			previousage=age
			previousyear=year
	
	writer.writerow(row)
	

""" Increment the id  of the runner for runners where we have
 same id,name and year of run same. Add the runner at the end of file"""
def IncrementId(row,writer):
	global idCount
	global last_id,yearsPerId
	
	id=int(row[0])
	year=int(row[7])
	InitializeCurrentId(id)
	
	if(id==currentId and idCount>0):
		for preYear in yearsPerId:
			if(preYear==year):
				last_id=last_id+1
				row[0]=last_id
	InitializeYear(year)
	#elif(id==currentId):
		#previousyear=year
	idCount=idCount+1
	writer.writerow(row)
	
""" Converts the time and pace into seconds"""
def ConvertingTimeIntoSeconds(row,writer):
	time=row[5]
	pace=row[6]
	hr, minutes, secs = [float(x) for x in time.split(':')]
	seconds=hr*3600 + minutes*60 + secs
	time=seconds
	row[5]=seconds

	min, sec = [float(x) for x in pace.split(':')]
	seconds= min*60 + sec
	row[6]=seconds
	
	Distance=26.219
	if(hr<2):
		time=row[6]*Distance
		row[5]=time
	writer.writerow(row)
	
""" Convert the sex column into numerical labels.
 I cross verified the labels with the labels given by labelencoder.
I assigned them here so that while loading the csv as numpy array,
 i dont have to do it for string columns and float columns separately """
def ConvertSexIntoLabels(row,writer):
	sex=row[3]
	if sex is "M":
		row[3]=1
	elif sex is'F':
		row[3]=2
	elif sex is'U':
		row[3]=0
	writer.writerow(row)

""" Call this function to apply all of the above preprocessings. Provide the file name, should have an extension mentioned."""
def PreProcessing(fileName):
	global currentId,previousage,previousyear,idCount,last_id
	with open(fileName, 'rb') as inp, open('output.csv', 'wb') as out:
		writer = csv.writer(out)
		for row in csv.reader(inp):
			if(row[0]!="Id"):
				RemoveZeroAge(row,writer)
				
	with open('output.csv', 'rb') as inp:
		fileObject=csv.reader(inp)
		last_id = sum(1 for row in fileObject) 
	
	with open('output.csv', 'rb') as inp,open('output1.csv', 'wb') as out:
		writer = csv.writer(out)
		for row in csv.reader(inp):
			if(row[0]!="Id"):
				IncrementAge(row,writer)
				
	os.remove('output.csv')
	currentId=0
	previousage=0
	previousyear=0
	idCount=0
	with open('output1.csv', 'rb') as inp,open('output2.csv', 'wb') as out:
		writer = csv.writer(out)
		for row in csv.reader(inp):
			if(row[0]!="Id"):
				IncrementId(row,writer)
				
	os.remove('output1.csv')
	currentId=0
	previousage=0
	previousyear=0
	idCount=0
	with open('output2.csv', 'rb') as inp,open('output3.csv', 'wb') as out:
		writer = csv.writer(out)
		for row in csv.reader(inp):
			if(row[0]!="Id"):
				ConvertingTimeIntoSeconds(row,writer)
	#os.remove('output2.csv')
	currentId=0
	previousage=0
	previousyear=0
	idCount=0
	with open('output3.csv', 'rb') as inp,open('output.csv', 'wb') as out:
		writer = csv.writer(out)
		for row in csv.reader(inp):
			if(row[0]!="Id"):
				ConvertSexIntoLabels(row,writer)
	os.remove('output3.csv')

