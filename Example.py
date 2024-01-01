from KNN_AlgoritmMachine_Learning import get_neighbors
from KNN_AlgoritmMachine_Learning import distance
from KNN_AlgoritmMachine_Learning import vote_distance_weights
train_set=[(1,2,2),(-3,-2,0),(1,1,3),(-3,-3,-2),(-3,-2,-0.5),(0,0.3,0.8),(-0.5,0.6,0.7),(0,0,0)]
labels=['apple','banana','apple','banana','apple',"orange",'orange','orange']
k=2
for test_instance in [(0,0,0),(2,2,2),(-3,-1,0),(0,1,0.9),(1,1.5,1.8),(0.9,0.8,1.6)]:
    neighbors=get_neighbors(train_set,labels,test_instance,k,distance=distance)
    print("Vote distance weights:",vote_distance_weights(neighbors))
