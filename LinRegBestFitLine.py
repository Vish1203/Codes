from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

xs = np.array([1,2,3,4,5,6], dtype=np.float64) #Defining numpy array and chaging data type to float for best fit.
ys = np.array([5,4,6,5,6,7], dtype=np.float64)

#Calculating y-intecept(b) and slope(m)
def best_fit_slope_and_interept(xs,ys):
    m = ( ((mean(xs) * mean(ys)) - mean(xs*ys)) /
        ((mean(xs)*mean(xs)) - mean(xs*xs))  )
    
    b = mean(ys) - m*mean(xs)
    return m,b

#For mean(xs)*mean(xs), we can also write it as mean(xs)**2 and similarly for
# mean(xs*xs) as mean(xs**2)

# The calculation follows the "PEMDAS" rule.!!!

m,b = best_fit_slope_and_interept(xs,ys)

#print(m,b)

regression_line = [(m*x)+b for x in xs] #regression_line means 'Y' that is being
#calculated using Y = MX+B

#The above line is used in place of:
#for x in xs:
    #regression_line.append((m*x)+b)

predict_x = 8 
predict_y = (m*predict_x)+b #if we want to predict the value of Y for given X =8

plt.scatter(xs,ys)
plt.scatter(predict_x,predict_y, color='g')
plt.plot(xs, regression_line)
plt.show() 








