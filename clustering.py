from linear_algebra import Vector,vector_mean,squared_distance
from typing import List
import tqdm
import itertools
import random
def num_differences(v1 : Vector , v2 : Vector) -> int:
    assert (len(v1) == len(v2))
    return len([x1 for x1,x2 in zip(v1,v2) if x1!=x2])

def cluster_means(k : int,
                  inputs : List[Vector],
                  assignments : List[int]) -> List[Vector]:
    clusters = [[] for i in range(k)]

    for input,assignments in zip(inputs,assignments):
        clusters[assignments].append(input)

    return [ vector_mean(cluster) if cluster else random.choice(inputs) for cluster in clusters]

class KMeans:
    def __init__ (self, k:int) -> None:
        self.k = k
        self.means = None
    
    def classify(self,input:Vector) -> int:
        return min(range(self.k), key = lambda i: squared_distance(input,self.means[i]))
    
    def train(self,inputs : List[Vector]) -> None:
        assignments = [random.randrange(self.k) for _ in inputs]

        with tqdm.trange(itertools.count()) as t:
            for _ in t:
                self.means = cluster_means(self.k,inputs,assignments)

                new_assignments = [self.classify(input) for input in inputs]

                num_changed = num_differences(assignments,new_assignments)
                if num_changed == 0:
                    return
                
                assignments = new_assignments
                self.means = cluster_means(self.k,inputs,assignments)
                t.set_description(f"changed: {num_changed} / {len(inputs)}")



image_path = r"girl_with_book.jpg"

import matplotlib.image as mpimg

img = mpimg.imread(image_path) / 256

top_row = img[0]
top_left_pixel = top_row[0]

red,green,blue = top_left_pixel

pixels = [pixel.tolist() for row in img for pixel in row]

clusterer = KMeans(5)
clusterer.train(pixels)

def recolor(pixels: Vector) -> Vector:
    cluster = clusterer.classify(pixels)
    return clusterer.means[cluster]

new_img = [[recolor(pixel) for pixel in row] for row in img]

import matplotlib.pyplot as plt


plt.imshow(new_img)
plt.axis('off')
plt.show()