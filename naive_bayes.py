import sst.load_sst
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


train_sst = sst.load_sst.get_train()
train_x = list(train_sst.keys())
train_y = list(train_sst.values())

test_sst = sst.load_sst.get_test()
test_x = list(test_sst.keys())
test_y = list(test_sst.values())

vec = CountVectorizer(stop_words='english')
train_x = vec.fit_transform(train_x).toarray()
test_x = vec.transform(test_x).toarray()
model = MultinomialNB()
model.fit(train_x, train_y)

print(model.score(test_x, test_y)) #accuracy: 0.3932126696832579
