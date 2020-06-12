import numpy as np
from sklearn import preprocessing, neighbors
from sklearn import model_selection
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data') #Reading the data file(csv) from local directory
df.replace('?', -99999, inplace=True)

#Replacing the ? (missing) values with -99999.
#We replace with -99999 because most algorithms recognise it as an outlier.

df.drop(['id'], 1, inplace=True)

#IF we do not drop ID, we get an accuracy of 0.6 which is absolutely wrong!
#We can also use df.dropna(inplace='True') in place of the above line.
#We are dropping id column as it is of no help to us in classifying breast cancer as benign or malignant.


#X-> Features and Y->Labels
X = np.array(df.drop(['class'], 1)) #Dropping class, as it is not a feature
Y = np.array(df['class'])

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, Y_train)

accuracy = clf.score(X_test, Y_test)
print(accuracy)

example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,2,2,2,3,2,1]])
#The numbers in the array do not occur in that order in the CSV data file so we are going to do a test on that
example_measures = example_measures.reshape(len(example_measures),-1) #For 2D arrays 
prediction = clf.predict(example_measures)
print(prediction)
