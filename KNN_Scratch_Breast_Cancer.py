import numpy as np
from math import sqrt
import warnings 
from collections import Counter
import pandas as pd
import random

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total votings groups!')
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])
            
        votes = [i[1] for i in sorted(distances)[:k]]
        #print(Counter(votes).most_common(1))
        #Counter(votes) creates a dictionary with number of times each element occurs in the list. most_common(1)
        #returns the element which occurs the most. 
        vote_result = Counter(votes).most_common(1)[0][0]
        confidence = Counter(votes).most_common(1)[0][1] / k

        #print(vote_result, confidence)
        
    return vote_result, confidence

df = pd.read_csv("breast-cancer-wisconsin.data")
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)
full_data = df.astype(float).values.tolist() #Convert datatype to float and list of lists
random.shuffle(full_data)

#slicing data
test_size = 0.2
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]} #2:emptylist
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):] #Everything upto last 20% data

#populate dictionaries:

for i in train_data:
    train_set[i[-1]].append(i[:-1])#Doung this to append list and make the attributes stay away from class that is the last column
  
for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote, confidence = k_nearest_neighbors(train_set, data, k=5)
        #k=5 because default doc of scitkilearn has k=5
        if group == vote:
            correct+=1
        else:
            print(confidence)
        total +=1
print('Accuracy:',correct/total)
