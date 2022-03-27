from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt
import pickle as pk
import time
import re
import numpy as np

def preprocess(list_of_sents):
    re_str = re.compile(r"(\b((n't)|(not)|(no)|(never))\s+)")
    return [re.sub(re_str, "\\1NOT_", sent) for sent in list_of_sents]

""" Preprocess for sentiment analysis https://web.stanford.edu/~jurafsky/slp3/4.pdf
1. binaryNB on the document level (remove duplicates in same review)
2. replace "didn't like" with "didn't "NOT"" """

train_x = preprocess(pk.load(open("./data/train_x.pk", "rb")))
train_y = pk.load(open("./data/train_y.pk", "rb"))

test_x = preprocess(pk.load(open("./data/test_x.pk", "rb")))
test_y = pk.load(open("./data/test_y.pk", "rb"))

t1 = time.time()

vec = TfidfVectorizer(stop_words='english')

train_x = vec.fit_transform(train_x).toarray()
train_x = np.where(train_x > 1, 1, train_x)
test_x = vec.transform(test_x).toarray()
test_x = np.where(test_x > 1, 1, test_x)
nb_model = MultinomialNB()
nb_model.fit(train_x, train_y)

# print(model.score(test_x, test_y)) #accuracy: 0.3932126696832579
pred_y = nb_model.predict(test_x)
print(classification_report(test_y, pred_y))

disp = ConfusionMatrixDisplay.from_estimator(
    nb_model,
    test_x,
    test_y,
    display_labels=list(range(1, 6)),
    cmap=plt.cm.Blues,
    normalize="true",
)
disp.ax_.set_title("Normalized Confusion Matrix: Naive Bayes")
print(disp.confusion_matrix)
plt.savefig("naivebayes_confusion.png")
plt.show()
print("Time elapsed NB: ", time.time() - t1)
