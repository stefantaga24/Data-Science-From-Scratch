from linear_algebra import Vector,distance
from typing import List
import tqdm
import random
def random_point(dim:int) -> Vector:
    return [random.random() for _ in range(dim)]

def random_distances(dim:int , num_pairs:int ) -> List[float]:
    p1 : Vector = random_point(dim)
    p2 : Vector = random_point(dim)
    return [distance(p1,p2) for _ in range(num_pairs)]

avg_distances : List[float] = []
min_distances : List[float]  = []
dimensions = range(1,100)
for dim in tqdm.tqdm(dimensions, desc= "The curse of dimensionality"):
    distances = random_distances(dim,10000)
    avg_distances.append(sum(distances)/10000)
    min_distances.append(min(distances))

from matplotlib import pyplot as plt


plt.plot(dimensions,avg_distances)

plt.show()


