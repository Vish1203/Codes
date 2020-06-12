from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

#xs = np.array([1,2,3,4,5,6], dtype=np.float64) #Defining numpy array and chaging data type to float for best fit.
#ys = np.array([5,4,6,5,6,7], dtype=np.float64)

def create_dataset(hm, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val+=step
        elif correlation and correlation == 'neg':
            val-=step
        xs = [i for i in range(len(ys))]
        
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

#hm=how many datapoints we want to create
#variance=hor variant we want the dataset to be
#step=how far on avg to step up the Y value per point
#Correlation will be true or false, for true, correlation should be positive and
#for that step should be positive and vice versa for negative correlation.


#Calculating y-intecept(b) and slope(m)
def best_fit_slope_and_intercept(xs,ys):
    m = ( ((mean(xs) * mean(ys)) - mean(xs*ys)) /
        ((mean(xs)*mean(xs)) - mean(xs*xs))  )
    
    b = mean(ys) - m*mean(xs)
    return m,b

#Calculates the e^2 or Squared error. Squared error is the error/distance bw best fit line and the data point in space.

def squared_error(ys_orig, ys_line):   
    return sum((ys_line - ys_orig)**2)

#Coeffiecient of determination i.e., r^2

def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 -(squared_error_regr / squared_error_y_mean)

xs, ys = create_dataset(40, 40, 2, correlation='pos') #Defining values for creating random dataset
#changing the value of variance to a smaller number gives larger coefficient of determination and vice versa
#changing correlation to false gives a very ugly dataset and a coefficient of determination as 0.00079500

 

m,b = best_fit_slope_and_intercept(xs,ys)

#print(m,b)

regression_line = [(m*x)+b for x in xs] #regression_line means 'Y' that is being
#calculated using Y = MX+B


predict_x = 8 
predict_y = (m*predict_x)+b #if we want to predict the value of Y for given X =8

r_squared = coefficient_of_determination(ys, regression_line)
print (r_squared)


plt.scatter(xs,ys)
plt.scatter(predict_x,predict_y, s=100, color='g')#s is size
plt.plot(xs, regression_line)
plt.show() 








