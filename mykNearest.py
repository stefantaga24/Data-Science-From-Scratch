from typing import NamedTuple,List
from linear_algebra import Vector,distance
from collections import Counter
import random
class LabeledPoint(NamedTuple):
    point : Vector
    label : str


def majority_vote(labels : List[str]) -> str :
    vote_counts = Counter(labels)
    winner, winner_count = vote_counts.most_common(1)[0]
    num_winners = len([count for count in vote_counts.values() if (count==winner_count)])
    if (num_winners==1):
        return winner
    else:
        return majority_vote(labels[:-1])
def knn_classify(k : int ,
                 labeled_points : List[LabeledPoint],
                 new_point: Vector) ->str:
    # Order the labeled points from nearest to farthest.
    by_distance = sorted(labeled_points,
                                key=lambda lp: distance(lp.point, new_point))
    # Find the labels for the k closest
    k_nearest_labels = [lp.label for lp in by_distance[:k]]
    # and let them vote.
    return majority_vote(k_nearest_labels)



from typing import Dict
 
import csv
from collections import defaultdict


def parse_iris_row(row : List[str]) -> LabeledPoint:
    measurements = [float(value) for value in row[:-1]]
    label = row[len(row)-1]

    return LabeledPoint(measurements,label)

with open("iris.dat") as f:
    reader = csv.reader(f)
    iris_data = [parse_iris_row(row) for row in reader]

points_by_species : Dict[str,List[Vector]] = defaultdict(list)

for iris in iris_data:
   points_by_species[iris.label].append(iris.point)


trainSet : List[LabeledPoint] = []
testSet : List[LabeledPoint] = []

for label in points_by_species:
    
    index = range(0,len(points_by_species[label]))
    index = list(index)
    random.shuffle(index)
    trainSetLim = int(len(points_by_species[label])/4)
    for i in range(0,trainSetLim):
        trainSet.append(LabeledPoint(points_by_species[label][index[i]],label))
    for i in range(trainSetLim+1,len(points_by_species[label])):
        testSet.append(LabeledPoint(points_by_species[label][index[i]],label))

for k in range(1,20):
    correct = 0 
    for point in testSet:
        correspondingLabel = knn_classify(k,trainSet,point.point)
        if (correspondingLabel == point.label):
            correct = correct+1

    print("Current K is : "  + str(k))
    accuracy  = correct/len(testSet)
    print("Accuracy is: " + str(accuracy))