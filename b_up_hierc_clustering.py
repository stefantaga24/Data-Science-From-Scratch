from typing import NamedTuple,Union,List,Callable
from linear_algebra import Vector
class Leaf(NamedTuple):
    value : Vector


leaf1 = Leaf([10,20])
leaf2 = Leaf([30,-15])

class Merged(NamedTuple):
    children: tuple
    order : int

merged = Merged((leaf1,leaf2),order=1)

Cluster = Union[Leaf,Merged]

def get_values(cluster:Cluster) -> List[Vector]:
    if isinstance(cluster, Leaf):
        return [cluster.value]
    else:
        return [value 
                for child in cluster.children
                for value in get_values(child)]
    
from linear_algebra import distance

def cluster_distance(cluster1:Cluster,
                     cluster2:Cluster,
                     distance_agg: Callable = min):
    return distance_agg([distance(v1,v2) for v1 in get_values(cluster1) for v2 in get_values(cluster2)])

def get_merge_order(cluster:Cluster)->float:
    if isinstance(cluster,Leaf):
        return float('inf')
    else:
        return cluster.order
    
def get_children(cluster : Cluster):
    if isinstance(cluster, Leaf):
        raise TypeError("Leaf has no children")
    else:
        return cluster.children
    
from typing import Tuple

def bottom_up_order(inputs: List[Vector],
                    distance_agg: Callable =min) ->Cluster:
    clusters : List[Cluster] = [Leaf(input) for input in inputs]

    def pair_distance(pair : Tuple[Cluster,Cluster]) -> float:
        return cluster_distance(pair[0], pair[1], distance_agg)
    
    while len(clusters) > 1 :
        c1,c2 = min(((cluster1,cluster2) for i,cluster1 in enumerate(clusters) for cluster2 in clusters[:i]), key = pair_distance)

        clusters = [c for c in clusters if c!=c1 and c!=c2]

        merged_cluster = Merged((c1,c2) , order = len(clusters))

        clusters.append(merged_cluster)

    return clusters[0]

