import sst.load_sst
import pickle as pk
import re

train_sst = sst.load_sst.get_train()
train_x = list(train_sst.keys())
train_y = list(train_sst.values())
pk.dump(train_x, open( "./data/train_x.pk", "wb" ))
pk.dump(train_y, open( "./data/train_y.pk", "wb" ))

test_sst = sst.load_sst.get_test()
test_x = list(test_sst.keys())
test_y = list(test_sst.values())
pk.dump(test_x, open( "./data/test_x.pk", "wb" ))
pk.dump(test_y, open( "./data/test_y.pk", "wb" ))
