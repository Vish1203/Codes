import pandas as pd
import quandl, math
import numpy as np
import sklearn import preprocessing

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
#makes changes to the 0.1 according to your need. 

#The abve stmt will predict out 10% of the data frame.

df['label'] = df[forecast_col].shift(-forecast_out)

#The above stmt is label. Label in Regression is output/prediction.


df.dropna(inplace=True)
print(df.head())


#The dropna() function is used to remove missing value

#print(df.tail())


 
