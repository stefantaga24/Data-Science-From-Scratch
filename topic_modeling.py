from typing import List
import random
from collections import Counter
def sample_from(weights : List[float]):
    total = sum(weights)

    rnd = total*random.random()
    for i,w in enumerate(weights):
        rnd -=w
        if rnd<=0: return i


# Draw 1000 times and count
draws = Counter(sample_from([0.1, 0.1, 0.8]) for _ in range(1000))
assert 10 < draws[0] < 190 # should be ~10%, this is a really loose test
assert 10 < draws[1] < 190 # should be ~10%, this is a really loose test
assert 650 < draws[2] < 950 # should be ~80%, this is a really loose test
assert draws[0] + draws[1] + draws[2] == 1000

documents = [
    ["Hadoop", "Big Data", "HBase", "Java", "Spark", "Storm", "Cassandra"],
    ["NoSQL", "MongoDB", "Cassandra", "HBase", "Postgres"],
    ["Python", "scikit-learn", "scipy", "numpy", "statsmodels", "pandas"],
    ["R", "Python", "statistics", "regression", "probability"],
    ["machine learning", "regression", "decision trees", "libsvm"],
    ["Python", "R", "Java", "C++", "Haskell", "programming languages"],
    ["statistics", "probability", "mathematics", "theory"],
    ["machine learning", "scikit-learn", "Mahout", "neural networks"],
    ["neural networks", "deep learning", "Big Data", "artificial intelligence"],
    ["Hadoop", "Java", "MapReduce", "Big Data"],
    ["statistics", "R", "statsmodels"],
    ["C++", "deep learning", "artificial intelligence", "probability"],
    ["pandas", "R", "Python"],
    ["databases", "HBase", "Postgres", "MySQL", "MongoDB"],
    ["libsvm", "regression", "support vector machines"]
]

K = 4

document_topic_counts = [Counter() for _ in documents]
topic_word_counts = [Counter() for _ in range(K)]

topic_counts = [0 for _ in range(K)]
document_lengths = [len(document) for document in documents]

distinct_words = set(word for document in documents for word in
document)
W = len(distinct_words)

D = len(documents)

def p_topic_given_document(topic: int, d: int, alpha: float = 0.1) -> float:
    """
    The fraction of words in document 'd'
    that are assigned to 'topic' (plus some smoothing)
    """
    return ((document_topic_counts[d][topic] + alpha) /
        (document_lengths[d] + K * alpha))

def p_word_given_topic(word: str, topic: int, beta: float = 0.1) -> float:
    """
    The fraction of words assigned to 'topic'
    that equal 'word' (plus some smoothing)
    """
    return ((topic_word_counts[topic][word] + beta) /
        (topic_counts[topic] + W * beta))


def topic_weight(d: int, word: str, k: int) -> float:
    """
    Given a document and a word in that document,
    return the weight for the kth topic
    """
    return p_word_given_topic(word, k) * p_topic_given_document(k, d)


def choose_new_topic(d: int, word: str) -> int:
    return sample_from([topic_weight(d, word, k)
                                    for k in range(K)])

random.seed(0)
document_topics = [[random.randrange(K) for word in document]
                        for document in documents]
for d in range(D):
    for word, topic in zip(documents[d], document_topics[d]):
        document_topic_counts[d][topic] += 1
        topic_word_counts[topic][word] += 1
        topic_counts[topic] += 1

