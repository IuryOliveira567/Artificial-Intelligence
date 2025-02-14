import math, random, statistics
import csv
import sys

sys.path.append('../utils')
from agent import Displayable
from utilities import argmax


boolean = [False, True]

class Data_set(Displayable):

    def __init__(self, train, test=None, prob_test=0.20, target_index=0,
                 header=None, target_type=None, one_hot=False,
                 seed=None):

        if(seed):
            random.seed(seed)

        if(test is None):
            train, test = partition_data(train, prob_test)

        self.train = train
        self.test = test

        self.display(1, "Training set has", len(train), "example. Number of columns: ", {len(e) for e in train})
        self.display(1, "Test set has", len(test), "examples. Number of columns: ", {len(e) for e in test})

        self.prob_test = prob_test
        self.num_properties = len(self.train[0])

        if(target_index < 0):
            self.target_index = self.num_properties + target_index
        else:
            self.target_index = target_index

        self.header = header
        self.domains = [set() for i in range(self.num_properties)]

        for example in self.train:
            for ind, val in enumerate(example):
                self.domains[ind].add(val)

        self.conditions_cache = {}
        self.create_features(one_hot)

        if(target_type):
            self.target_ftype = target_type

        self.display(1, "There are", len(self.input_features), "input features")

    def __str__(self):

        if(self.train and len(self.train) > 0):
            return("Data: " + str(len(self.train)) + " training examples, "
                   + str(len(self.test)) + " test examples, "
                   + str(len(self.train[0])) + " features.")
        else:
            return("Data: " + str(len(self.train)) + " training examples, "
                   + str(len(self.test)) + "test examples.")

    def create_features(self, one_hot=False):

        self.target = None
        self.input_features = []

        for i in range(self.num_properties):
            frange = list(self.domains[i])
            ftype = self.infer_type(frange)

            if(one_hot and ftype == "categorical" and i != self.target_index):
                for val in frange:
                    def feat(e, index=i, val=val):
                        return e[index] == val

                    if(self.header):
                        feat.__doc__= self.header[i] + "=" + val
                    else:
                        feat.__doc__ = f"e[{i}]={val}"

                    feat.frange = boolean
                    feat.type = "boolean"
                    self.input_features.append(feat)
            else:
                def feat(e, index=i):
                    return e[index]

                if(self.header):
                    feat.__doc__ = self.header[i]
                else:
                    feat.__doct__ = "e[" + str(i) + "]"

                feat.frange = frange
                feat.ftype = ftype

                if(i == self.target_index):
                    self.target = feat
                else:
                    self.input_featres.append(feat)

    def infer_type(self, domain):

        if(all(v in {True, False} for v in domain) or all(v in {0, 1} for v in domain)):
            return "boolean"

        if(all(isinstance(v, (float, int)) for v in domain)):
            return "numeric"
        else:
            return "categorical"
        
        

























        
