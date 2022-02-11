from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle as pk
import time
train_x = pk.load(open( "./data/train_x.pk", "rb" ))
train_y = pk.load(open( "./data/train_y.pk", "rb" ))

test_x = pk.load(open( "./data/test_x.pk", "rb" ))
test_y = pk.load(open( "./data/test_y.pk", "rb" ))

t1=time.time()
vec = CountVectorizer(stop_words='english')
train_x = vec.fit_transform(train_x).toarray()
test_x = vec.transform(test_x).toarray()
nb_model = MultinomialNB()
nb_model.fit(train_x, train_y)

#print(model.score(test_x, test_y)) #accuracy: 0.3932126696832579
pred_y = nb_model.predict(test_x)
print(classification_report(test_y, pred_y))

disp = ConfusionMatrixDisplay.from_estimator(
    nb_model,
    test_x,
    test_y,
    display_labels=list(range(1,6)),
    cmap=plt.cm.Blues,
    normalize="true",
)
disp.ax_.set_title("Normalized Confusion Matrix: Naive Bayes")
print(disp.confusion_matrix)
plt.savefig("naivebayes_confusion.png")
plt.show()
print("Time elapsed NB: ", time.time() - t1)
