import numpy as np 
from sklearn import datasets 
iris=datasets.load_iris()
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 
from collections import Counter 


data=iris.data
labels=iris.target 
for i in[0,79,99,101]:
    print(f"index:{i:3},features:{data[i]},label:{labels[i]}")
#we will have to create a train set, using permutation from np.random to split the data randomly
np.random.seed(42)
indicies=np.random.permutation(len(data))
n_training_samples=12
learn_data=data[indicies[:-n_training_samples]]
learn_labels=labels[indicies[:-n_training_samples]]
test_data=data[indicies[-n_training_samples:]]
test_labels=labels[indicies[n_training_samples:]]        
print("The first samples of our learn_set:")
print(f"{'index':7s}{'data':20s}{'labels':3s}")
for i in range(5):
    print(f"{i:4d}{learn_data[i]}{learn_labels[i]:3}")
colors=("r","b")
X=[]
for iclass in range(3):
    X.append([[],[],[]])
    for i in range(len(learn_data)):
        if learn_labels[i]==iclass:
            X[iclass][0].append(learn_data[i][0])
            X[iclass][1].append(learn_data[i][1])
            X[iclass][2].append(sum(learn_data[i][2:]))
colors=("r","g","y")
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
for iclass in range(3):
    ax.scatter(X[iclass][0],X[iclass][1],X[iclass][2],c=colors[iclass])
plt.show()
import numpy as np 

#function to calculate the Eucladian and pythagorean distances

def distance(instance1,instance2):
    return np.linalg.norm(np.subtract(instance1,instance2))
print(distance([3,5],[1,1]))
print(distance(learn_data[3],learn_data[44])) 

#function to get the nearest neighbours 

def get_neighbors(training_set,labels,test_instance,k,distance):
    distances=[]
    for index in range(len(training_set)):
        dist=distance(test_instance,training_set[index])
        distances.append((training_set[index],dist,labels[index]))
        distances.sort(key=lambda x:x[1])
        neighbors=distances[:k]
    return neighbors
for i in range(5):
    neighbors=get_neighbors(learn_data,learn_labels,test_data[i],3,distance=distance)
    print("Index:     ",i,'\n',"Testset Data: ",test_data[i],'\n',"Testset Label:",test_labels[i],'\n',"Neighbors: ",neighbors,'\n')

def vote(neighbors):
    class_counter=Counter()
    for neighbor in neighbors:
        class_counter[neighbor[2]] +=1
    return class_counter.most_common(1)[0][0]
for i in range(n_training_samples):
    neighbors=get_neighbors(learn_data,learn_labels,test_data[i],3,distance=distance)
    print("index: ",i,",result of vote:",vote(neighbors),",label:",test_labels[i],",data:",test_data[i])
def vote_prob(neighbors):
    class_counter=Counter()
    for neighbor in neighbors:
        class_counter[neighbor[2]] +=1
    labels,votes=zip(*class_counter.most_common())
    winner=class_counter.most_common(1)[0][0]
    votes4winner=class_counter.most_common(1)[0][1]
    return winner,votes4winner/sum(votes)
for i in range(n_training_samples):
    neighbors=get_neighbors(learn_data,learn_labels,test_data[i],5,distance=distance)
    print("index: ",i,", vote_prob: ",vote_prob(neighbors),",label: ",test_labels[i],",data: ",test_data[i])
              
#Lets consider that we have an unknown object and 11 neighbours, the closest neighbours are 5 class A nad the furthest neighbours are 6 of class B, from our latter method the UO, would be classified as B object , which is not true since the Class B is furthest from the UO
#To solve this we will assign weights to the neighbours using the harmonc series, 
def vote_harmonic_weights(neighbors,all_results=True):
    class_counter=Counter()
    number_of_neighbors=len(neighbors)
    for index in range(number_of_neighbors):
        class_counter[neighbors[index][2]]+=1/(index+1)
    labels,votes=zip(*class_counter.most_common())
    winner=class_counter.most_common(1)[0][0]
    votes4winner=class_counter.most_common(1)[0][1]
    if all_results:
        total=sum(class_counter.values(),0.0)
        for key in class_counter:
            class_counter[key] /=total
            return winner,class_counter.most_common()
        else:
            return winner, votes4winner/sum(votes)
for i in range(n_training_samples):
    neighbors=get_neighbors(learn_data,learn_labels,test_data[i],6,distance=distance)
    print("Index:  ",i,",result of vote: ",vote_harmonic_weights(neighbors,all_results=True))

def vote_distance_weights(neighbors,all_results=True):
    class_counter=Counter()
    number_of_neighbors=len(neighbors)
    for index in range(number_of_neighbors):
        dist=neighbors[index][1]
        label=neighbors[index][2]
        class_counter[label]+=1/(dist**2 +1)
    labels,votes=zip(*class_counter.most_common())
    winner=class_counter.most_common(1)[0][0]
    votes4winner=class_counter.most_common(1)[0][1]
    if all_results:
        total=sum(class_counter.values(),0.0)
        for key in class_counter:
            class_counter[key]/=total
            return winner, class_counter.most_common()
        else:
            return winner, votes4winner/sum(votes)
for i in range(n_training_samples):
    neighbors=get_neighbors(learn_data,learn_labels,test_data[i],6,distance=distance)
    print("index: ",i,vote_distance_weights(neighbors,all_results=True))
                    

    

