import matplotlib.pyplot as plt
import pandas as pd
import pickle as pk

train_x = pk.load(open( "./data/train_x.pk", "rb" ))
train_y = pk.load(open( "./data/train_y.pk", "rb" ))

test_x = pk.load(open( "./data/test_x.pk", "rb" ))
test_y = pk.load(open( "./data/test_y.pk", "rb" ))

df_test_y = pd.DataFrame(test_y) + 1
df_train_y = pd.DataFrame(train_y) + 1
df_test_x = pd.DataFrame(test_x)

# print(df_train_y[0].unique(), df_test_y[0].unique())
# print(df_train_y.mode() + 1)
# print(df_test_y.mode() + 1)
# print(df_test_y.median())

plt.style.use('ggplot')
df_train_y.value_counts().plot(kind = "barh")
plt.ylabel("Sentiment Rating")
plt.xlabel("Number of Sentences")
plt.title("Sentiment Rating Distribution In Training Set", loc = "left")
plt.show()