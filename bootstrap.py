from typing import TypeVar, Callable,List
import random
X = TypeVar('X')
Stat = TypeVar('Stat')

def bootstrap_sample(data : List[X]) -> List[X]:
    return [random.choice(data) for _ in data]

def bootstrap_statistic(data : List[X],
                        stats_fn: Callable[[List[X]],Stat],
                        num_samples:int) -> List[Stat]:
    return [stats_fn(bootstrap_sample(data)) for _ in range(num_samples)]

 