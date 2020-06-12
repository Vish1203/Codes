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

style.use('ggplot')


df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df ['Adj. Close'] *100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df ['Adj. Open'] *100.0

#           price        x         x           x     
df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)
forecast_out = int(math.ceil(0.1*len(df)))
#print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)


X = np.array(df.drop(['label', 'Adj. Close'],1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
 #X_lately is the varible we are going to predict against.

df.dropna(inplace=True)
Y = np.array(df['label'])
Y = np.array(df['label'])

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2)

clf = LinearRegression() #Classifier
clf.fit(X_train, Y_train) #fit is synonymous with train
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










 
