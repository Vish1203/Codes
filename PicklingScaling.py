import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, svm
from sklearn import model_selection
#from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import style
import pickle 


#Pickle is serialization of any python object like classifier. Pickling is basically
#like opening a file and saving it and then opening it again to read it.
#Pickling is done to remove the training of data each time data comes in as training is tedious.
#Pickling is saving time by not training the data each time we need to make a new prediction.

style.use('ggplot')

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df ['Adj. Close'] *100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df ['Adj. Open'] *100.0

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)
forecast_out = int(math.ceil(0.01*len(df)))
#print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X = X[:-forecast_out]
X_lately = X[-forecast_out:] #X_lately is the varible we are going to predict against.

df.dropna(inplace=True)
Y = np.array(df['label'])

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2)

clf = LinearRegression() #Classifier
clf.fit(X_train, Y_train) #fit is synonymous with train

#Pickling:
#After Pickling is successfully executed, a file is created in the folder that holds your python files.
#Here, the pickle file is names linearregression.pickle

with open('linearregression.pickle' , 'wb') as f:
    pickle.dump(clf, f) #Dumps the classifier at f

pickle_in = open('linearregression.pickle' , 'rb')
clf = pickle.load(pickle_in) #Redefined classifier here.!

#The “r” means to just read the file. You can also open a file in “rb” (read binary), “w” (write), “a” (append), or “wb” (write binary).
#Note that if you use either “w” or “wb”, Python will overwrite the file, if it exists already or create it if the file doesn't exist.

accuracy = clf.score(X_test, Y_test) #score is synonymous with test

forecast_set = clf.predict(X_lately)
print(forecast_set, accuracy, forecast_out)
df['Forecast'] =np.nan #entire column is full of not a number data

last_date = df.iloc[-1].name #gets the last date; -1 means the last date
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

#Below: We are itearating through the forecast set. Taking each Forecast and day
#and then setting those as the values in the data frame.
#the future feature is not a number
#This loop is defined to get dates on the axes on the graph
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] +[i] #.loc references the index in data frame.
# The above line takes all of the first columns and sets them to not a number and
#the final column is whatever 'i' is, which is the forecast in this case.

print(df.tail())

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()










 
