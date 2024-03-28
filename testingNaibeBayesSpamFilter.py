import glob,re 
from typing import List 
from naiveBayesSpamFilter import Message,NaiveBayesClassifier
path = 'spam_data/*/*'

from typing import TypeVar, List, Tuple
import random
X = TypeVar('X') # generic variable type


def split_data(data: List[X], prob: float) -> Tuple[List[X], List[X]]:
    """Split data into fractions [prob, 1 - prob]"""
    data = data[:] # Make a shallow copy
    random.shuffle(data) # because shuffle modifies the list.
    cut = int(len(data) * prob) # Use prob to find a cutoff
    return data[:cut], data[cut:] # and split the shuffled list there.

data : List[Message] = []

for filename in glob.glob(path):
    is_spam = "ham" not in filename

    with open(filename, errors = 'ignore') as email_file:
        for line in email_file:
            if (line.startswith("Subject:")):
                subject = line.lstrip("Subject: ")
                data.append(Message(subject,is_spam)) 
                break

random.seed(0)
train_messages, test_messages = split_data(data,0.75)

model = NaiveBayesClassifier(min_frequency = 1)
model.train(train_messages)

from collections import Counter

predictions = [(message,model.predict(message.text)) for message in test_messages]

confusion_matrix = Counter((message.is_spam, spam_probability > 0.5) for message,spam_probability in predictions)

print(confusion_matrix)
print("precision is:" + str(confusion_matrix[(True,True)] / (confusion_matrix[(True,True)] + confusion_matrix[(False,True)])))
print("recall is:" + str(confusion_matrix[(True,True)] / (confusion_matrix[(True,True)] + confusion_matrix[(True,False)])))