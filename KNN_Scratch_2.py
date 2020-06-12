import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings #For warning the user of inappropriate "K" values
from matplotlib import style
from collections import Counter
style.use('fivethirtyeight')

dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]} #k and r are classes and the inputs are features
new_features = [5,7]   #New datapoint


# OR you use this instead of the above line
 # for i in dataset:
    # for ii in dataset[i]: #ii means the data pair like 1,2
       # [plt.scatter(ii[0],ii[1], s=100, color=i)]


def k_nearest_neighbors(data, predict, k=3):
    if len(data) >=k:
        warnings.warn('K is set to a value less than total votings groups!')
    distances = []
    for group in data: #iterates through the dataset and its classes k and r
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])
            #The above way of calculating distance is faster so we use it instead of the two written below
            #euclidean_distance = sqrt( (features[0]-predict[0])**2 + (features[1]-predict[1]**2 ) )
            #Calculating the distance using above line is not very efficient and it is a little slow for large datasets and thisone is just hardcoded for 2D data. What if the data was 3D?
            #euclidean_distance = np.sqrt(np.sum((np.array(features)-np.array(predict))**2))
            #We are not using the np way of calculating distance either because numpy has an even easier way to calculate euclidean distance
        
        votes = [i[1] for i in sorted(distances)[:k]] #Keeps only the top 3 distances for final comparsion as k=3
        print(Counter(votes).most_common(1)) #Counts the first most common vote
        vote_result = Counter(votes).most_common(1)[0][0]
        
    return vote_result

result =  k_nearest_neighbors(dataset, new_features, k=3)
print(result)

[[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_features[0], new_features[1], s=100, color=result)
plt.show()
