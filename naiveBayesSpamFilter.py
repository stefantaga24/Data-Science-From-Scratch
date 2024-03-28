from typing import Set
import re

def tokenize(text : str) -> Set[str]:
    text = text.lower()
    all_words = re.findall("[a-z0-9]+", text)
    return set(all_words)


assert tokenize("Data Science is science") == {"data" , "science", "is"}

from typing import NamedTuple

class Message(NamedTuple):
    text : str
    is_spam : bool


from typing import List, Tuple , Dict , Iterable
import math
from collections import defaultdict

class NaiveBayesClassifier: 
    def __init__ (self, k: float = 0.5, min_frequency = 20) -> None:
        self.k = k

        self.tokens : Set[str] = set()
        self.token_frequency : Dict[str,int] = defaultdict(int)
        self.token_spam_counts : Dict[str,int] = defaultdict(int) #spam messages that contain a certain token
        self.token_ham_counts : Dict[str,int] = defaultdict(int) # ham messages that contain a certain token
        self.spam_messages = self.ham_messages = 0
        self.curated = 0
        self.min_frequency = min_frequency

    def train(self, messages: Iterable[Message]):
        for message in messages:
            if (message.is_spam):
                self.spam_messages +=1
            else:
                self.ham_messages +=1

            for token in tokenize(message.text):
                self.tokens.add(token)
                self.token_frequency[token]+=1
                if (message.is_spam):
                    self.token_spam_counts[token] = self.token_spam_counts[token]+1
                else:
                    self.token_ham_counts[token] = self.token_ham_counts[token]+1

    def _probabilities(self, token:str) -> Tuple[float,float]:
        """ returns P(spam|token) and P(ham|token) """

        probability_token_spam : float  = (self.k+self.token_spam_counts[token]) / (2*self.k + self.spam_messages)
        probability_token_ham : float  = (self.k+self.token_ham_counts[token]) / (2*self.k + self.ham_messages)

        return probability_token_spam,probability_token_ham
    

    def curateTokens(self):
        new_tokens = []
        for token in self.tokens:
            if (self.token_frequency[token] > self.min_frequency):
                new_tokens.append(token)
            else:
                print(token)
        self.tokens = new_tokens

    def predict(self, text:str) -> float:
        """ probability of the message to be spam"""
        if (self.curated ==0):
            self.curateTokens()
        spam_probability = 0
        ham_probability = 0
        text_tokens = tokenize(text)
        for token in self.tokens:
            prob_if_spam, prob_if_ham = self._probabilities(token)
            if token in text_tokens:
                spam_probability += math.log(prob_if_spam)
                ham_probability += math.log(prob_if_ham)
            else:
                spam_probability += math.log(1.0 - prob_if_spam)
                ham_probability += math.log(1.0 - prob_if_ham)

        
        spam_probability = math.exp(spam_probability)
        ham_probability = math.exp(ham_probability)

        return spam_probability/ (spam_probability + ham_probability)
    
messages = [Message("spam rules", is_spam=True),
Message("ham rules", is_spam=False),
Message("hello ham", is_spam=False)]
model = NaiveBayesClassifier(k=0.5)
model.train(messages)

assert model.tokens == {"spam", "ham", "rules", "hello"}
assert model.spam_messages == 1
assert model.ham_messages == 2
assert model.token_spam_counts == {"spam": 1, "rules": 1}
assert model.token_ham_counts == {"ham": 2, "rules": 1, "hello": 1}

text = "hello spam"

print(model.predict(text))