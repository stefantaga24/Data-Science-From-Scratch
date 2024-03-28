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
 

from matplotlib import pyplot as plt

metrics = ['sepal length', 'sepal width', 'petal length', 'petal width']
pairs = [(i, j) for i in range(4) for j in range(4) if i < j]
marks = ['+', '.', 'x'] # we have 3 classes, so 3 markers

fig, ax = plt.subplots(2, 3)
for row in range(2):
    for col in range(3):
        i, j = pairs[3 * row + col]
        ax[row][col].set_title(f"{metrics[i]} vs {metrics[j]}", fontsize=8)
        ax[row][col].set_xticks([])
        ax[row][col].set_yticks([])
        for mark, (species, points) in zip(marks, points_by_species.items()):
            xs = [point[i] for point in points]
            ys = [point[j] for point in points]
            ax[row][col].scatter(xs, ys, marker=mark, label=species)
            ax[-1][-1].legend(loc='lower right', prop={'size': 6})
plt.show()

from typing import TypeVar, List, Tuple
X = TypeVar('X') # generic type to represent a data point
def split_data(data: List[X], prob: float) -> Tuple[List[X], List[X]]:
    """Split data into fractions [prob, 1 - prob]"""
    data = data[:] # Make a shallow copy
    random.shuffle(data) # because shuffle modifies the list.
    cut = int(len(data) * prob) # Use prob to find a cutoff
    return data[:cut], data[cut:] # and split the shuffled list there.

import random
random.seed(12)
iris_train, iris_test = split_data(iris_data, 0.70)
assert len(iris_train) == 0.7 * 150
assert len(iris_test) == 0.3 * 150

from typing import Tuple
# track how many times we see (predicted, actual)
confusion_matrix: Dict[Tuple[str, str], int] = defaultdict(int)
num_correct = 0
for iris in iris_test:
    predicted = knn_classify(5, iris_train, iris.point)
    actual = iris.label
    if predicted == actual:
        num_correct += 1
    confusion_matrix[(predicted, actual)] += 1
pct_correct = num_correct / len(iris_test)
print(pct_correct, confusion_matrix)
