from statistics import mean
import numpy as np
import matplotlib.pyplot as plt

xs = np.array([1,2,3,4,5,6], dtype=np.float64) #Defining numpy array and chaging data type to float for best fit.
ys = np.array([5,4,6,5,6,7], dtype=np.float64)

def best_fit_slope(xs,ys):
    m = ( ((mean(xs) * mean(ys)) - mean(xs*ys)) /
        ((mean(xs)*mean(xs)) - mean(xs*xs))  ) 
    return m

#For mean(xs)*mean(xs), we can also write it as mean(xs)**2 and similarly for
# mean(xs*xs) as mean(xs**2)

# The calculation follows the "PEMDAS" rule.!!!

m = best_fit_slope(xs,ys)

print(m)








