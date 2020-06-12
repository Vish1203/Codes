import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing, svm
from sklearn import model_selection
#from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
#The next line calculates the low high percentage
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df ['Adj. Close'] *100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df ['Adj. Open'] *100.0

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

#The above line has features. Features in Regression are inputs.

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

#The above stmt fills the absent values with -99999 and in python
#In ML, we cant work with NaN (Not a number) data so it should be replaced, rather than getting rid of data.

#ceil gives the upper value and floor gives the lower value

forecast_out = int(math.ceil(0.01*len(df)))
print(forecast_out)
#makes changes to the 0.1 according to your need. 

#The abve stmt will predict out 10% of the data frame.

df['label'] = df[forecast_col].shift(-forecast_out)

#The above stmt is label. Label in Regression is output/prediction.


df.dropna(inplace=True)

#The dropna() function is used to remove missing value
#print(df.tail()) This is used to get last few values in the column.

#cross_validation is used to  create training and testing samples.
#cross_validation is used to split up the data and separate the data.
#Cross validation is deprecated, so we use model_selection instead of Cross_validation.!

#Defining X and Y ie. Features and labels (respectively) to train and test data.

X = np.array(df.drop(['label'],1))

#We are dropping the 'label' column here.

Y = np.array(df['label'])

#Scaling before feeding it to the classifier:-

X = preprocessing.scale(X)

#X = X[:-forecast_out+1]
#Remove this because we already dropped the na columns
#This is a case of slice indexing, and also uses negative indices.
#It means that the last forecast_out + 1 elements of X are discared.
#For example,>>> my_list = [1, 2, 3, 4, 5]
#>>> my_list[:-2]
#[1, 2, 3]

Y = np.array(df['label'])

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2)
#0.2 means we want to use 20% of the data as testing data. The test_split shuffles the data
#and put it into X_train, X_test, Y_train, Y_test.

#Classifier being used is Linear Regression:

clf = LinearRegression() #Classifier
clf.fit(X_train, Y_train) #fit is synonymous with train
accuracy = clf.score(X_test, Y_test) #score is synonymous with test

#Here, Accuracy is, of what the price would be shifted 1% of the days.

print(accuracy)

#Accuracy attained is 0.9798203576833021 ie almost 98%







 
