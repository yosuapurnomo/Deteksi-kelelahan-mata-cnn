import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold

class crossValidation:

    def __init__(self, data, label):
        self.data = data
        self.label = label

    def splitData(self, nSplit):
        kf = KFold(n_splits=nSplit)
        X = []
        for train_index, test_index in kf.split(self.data, self.label):
            X.append([np.array(train_index), np.array(test_index)])
        return X