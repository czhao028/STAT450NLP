from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import pickle as pk
train_x = pk.load(open( "./data/train_x.pk", "rb" ))
train_y = pk.load(open( "./data/train_y.pk", "rb" ))

test_x = pk.load(open( "./data/test_x.pk", "rb" ))
test_y = pk.load(open( "./data/test_y.pk", "rb" ))
import time
t1 = time.time()
pipeline = Pipeline(
    [
        ('vect', CountVectorizer(stop_words="english")),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(
            loss='hinge',
            penalty='l2',
            alpha=1e-3,
            random_state=42,
            max_iter=100,
            learning_rate='optimal',
            tol=None,
        )),
])

svm_model = pipeline.fit(train_x, train_y)
pred_y = svm_model.predict(test_x)

print(classification_report(test_y, pred_y))

disp = ConfusionMatrixDisplay.from_estimator(
    svm_model,
    test_x,
    test_y,
    display_labels=list(range(1,6)),
    cmap=plt.cm.Blues,
    normalize="true",
)
disp.ax_.set_title("Normalized Confusion Matrix: SVM")
print(disp.confusion_matrix)
plt.savefig("svm_confusion.png")
plt.show()

print("Time elapsed SVM: ", time.time() - t1) #Time elapsed SVM:  2.7469582557678223