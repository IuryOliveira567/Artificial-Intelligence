import math, random, statistics
import csv
import sys

sys.path.append('../utils')
from agent import Displayable
from utilities import argmax


boolean = [False, True]

class Data_set(Displayable):

    def __init__(self, train, test=None, prob_test=0.20, target_index=-1,
                 header=None, target_type=None, one_hot=False,
                 seed=None):

        if(seed):
            random.seed(seed)

        if(test is None):
            train, test = partition_data(train, prob_test)

        self.train = train
        self.test = test

        self.display(1, f"Training set has", len(train), "example. Number of columns: ", {len(e) for e in train})
        self.display(1, f"Test set has", len(test), "examples. Number of columns: ", {len(e) for e in test})

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
                    feat.__doc__ = "e[" + str(i) + "]"

                feat.frange = frange
                feat.ftype = ftype

                if(i == self.target_index):
                    self.target = feat
                else:
                    self.input_features.append(feat)

    def infer_type(self, domain):

        if(all(v in {True, False} for v in domain) or all(v in {0, 1} for v in domain)):
            return "boolean"
        if(all(isinstance(v, (float, int)) for v in domain)):
            return "numeric"
        else:
            return "categorical"

    def conditions(self, max_num_cuts=8, categorical_only=False):

        if((max_num_cuts, categorical_only) in self.conditions_cache):
            return self.conditions_cache[(max_num_cuts, categorical_only)]

        conds = []

        for ind, frange in enumerate(self.domains):
            if(ind != self.target_index and len(frange) > 1):
                if(len(frange) == 2):
                    true_val = list(frange)[1]

                    def feat(e, i=ind, tv=true_val):
                        return e[i] == tv

                    if(self.header):
                        feat.__doc__ = f"{self.header[ind]} == {true_val}"
                    else:
                        feat.__doc__ = f"e[{ind}] == {true_val}"

                    feat.frange = boolean
                    feat.ftype = "boolean"

                    conds.append(feat)
            elif(all(isinstance(val, (int, float)) for val in frange)):
                if(categorical_only):
                    def feat(e, i=ind):
                        return e[i]

                    feat.__doc__ = f"e[{ind}]"
                    conds.append(feat)
                else:
                    sorted_frange = sorted(frange)
                    num_cuts = min(max_num_cuts, len(frange))

                    cut_positions = [len(frange) * i // num_cuts for i in range(1, num_cuts)]

                    for cut in cut_positions:
                        cut_at = sorted_frange[cut]
                        def feat(e, ind_=ind, cutat=cut_at):
                            return e[ind_] < cut_at

                        if(self.header):
                            feat.__doc__ = self.header[ind] + "<" + str(cut_at)
                        else:
                            feat.__doc__ = "e[" + str(ind) + "]<" + str(cut_at)

                        feat.frange = boolean
                        feat.ftype = "boolean"
                        conds.append(feat)
            else:
                for val in frange:
                    def feat(e, ind_=ind, val_=val):
                        return e[ind_] == val_

                    if(self.header):
                        feat.__doc__ = self.header[ind] + "==" + str(val)
                    else:
                        feat.__doc__ = "e[" + str(ind) + "]==" + str(val)

                    feat.frange = boolean
                    feat.ftype = "boolean"

                    conds.append(feat)

        self.conditions_cache[(max_num_cuts, categorical_only)] = conds
        return conds

    def evaluate_dataset(self, data, predictor, error_measure, label=-1):

        if(data):
            try:
                value = statistics.mean(
                    error_measure(
                        predictor(e[:label]),
                        self.target(e)
                    )
                    for e in data)
            except ValueError:
                return float("inf")
            return value
        else:
            return math.nan

class Evaluate(object):

    def squared_loss(self, prediction, actual):
        "squared loss"

        if(isinstance(prediction, (list, dict))):
            return (1 - prediction[actual]) ** 2
        else:
            return (prediction - actual) ** 2
 
    def absolute_loss(self, prediction, actual):
        "absolute loss"
        if(isinstance(prediction, (list, dict))):
            return abs(1 - prediction[actual])
        else:
            return abs(prediction - actual)

    def log_loss(self, prediction, actual):
        "log loss"        
        try:
            if(isinstance(prediction, (list, dict))):
                return -math.log2(prediction[actual])
            else:
                return -math.log2(prediction) if actual == 1 else -math.log2(1 - prediction)
        except ValueError:
            return float("inf")

    def accuracy(self, prediction, actual):
        "accuracy"
        if(isinstance(prediction, dict)):
          prev_val = prediction[actual]
          return 1 if all(prev_val >= v for v in prediction.values()) else 0
        else:
          return 1 if abs(actual - prediction) <= 0.5 else 0
    
    all_criteria = [accuracy, absolute_loss, squared_loss, log_loss]


def partition_data(data, prob_test=0.30):

    train = []
    test = []

    for example in data:
        if(random.random() < prob_test):
            test.append(example)
        else:
            train.append(example)

    return train, test

class Data_from_file(Data_set):

    def __init__(self, file_name, separator=',', num_train=None,
                 prob_test=0.3, has_header=False, target_index=-1, one_hot=False,
                 categorical=[], target_type=None, include_only=None, seed=None):

        with open(file_name, 'r', newline='') as csvfile:
            self.display(1, "Loading", file_name)
            data_all = (line.strip().split(separator) for line in csvfile)

            if(include_only is not None):
                data_all = ([v for (i, v) in enumerate(line) if i in include_only]
                            for line in data_all)

            if(has_header):
                header = next(data_all)
            else:
                header = None

            data_tuples = (interpret_elements(d) for d in data_all if len(d) > 1)

            if(num_train is not None):
                train = []

                for i in range(num_train):
                    train.append(next(data_tuples))

                test = list(data_tuples)
                Data_set.__init__(self, train, test=test,
                                  target_index=target_index, header=header)
            else:
                Data_set.__init__(self, data_tuples, test=None,
                                  prob_test=prob_test, target_index=target_index,
                                  header=header, seed=seed, target_type=target_type,
                                  one_hot=one_hot)

class Data_from_files(Data_set):

    def __init__(self, train_file_name, test_file_name, separator=',',
                 has_header=False, target_index=0, one_hot=False,
                 categorical=[], target_type=None, include_only=None):

        with open(train_file_name, 'r', newline='') as train_file:
            with open(test_file_name, 'r', newline='') as test_file:
                train_data = (line.strip().split(separator) for line in train_file)
                test_data = (line.strip().split(separator) for line in test_file)

                if(include_only is not None):
                    train_data = ([v for (i, v) in enumerate(line) if i in include_only]
                                  for line in train_data)

                    test_data = ([v for (i, v) in enumerate(line) if i in include_only]
                                 for line in test_data)

                if(has_header):
                    header = next(train_data)
                else:
                    header = None

                train_tuples =[interpret_elements(d) for d in train_data if(len(d) > 1)]
                test_tuples = [interpret_elements(d) for d in test_data if(len(d) > 1)]

                Data_set.__init__(self, train_tuples, test_tuples, target_index=target_index, header=header,
                                  one_hot=one_hot)

def interpret_elements(str_list):

    res = []
    
    for e in str_list:
        try:
            res.append(int(e))
        except ValueError:
            try:
                res.append(float(e))
            except ValueError:
                se = e.strip()

                if(se in ["True", "true", "TRUE"]):
                    res.append(True)
                elif(se in ["False", "false", "FALSE"]):
                    res.append(False)
                else:
                    res.append(e.strip())

    return res

class Data_set_augmented(Data_set):

    def __init__(self, dataset, unary_functions=[], binary_functions=[],
                 include_orig=True):

        self.orig_dataset = dataset
        self.unary_functions = unary_functions
        self.binary_functions = binary_functions
        self.include_orig = include_orig

        self.target = dataset.target
        Data_set.__init__(self, dataset.train, test=dataset.test,
                          target_index=dataset.target_index)

    def create_features(self, one_hot=False):

        if(self.include_orig):
            self.input_features = self.orig_dataset.input_features.copy()
        else:
            self.input_features = []

        for u in self.unary_functions:
            for f in self.orig_dataset.input_features:
                self.input_features.append(u(f))

        for b in self.binary_functions:
            for f1 in self.orig_dataset.input_features:
                for f2 in self.orig_dataset.input_features:
                    if(f1 != f2):
                        self.input_features.append(b(f1, f2))

def square(f):

    def sq(e):
        return f(e) ** 2

    sq.__doc__ = f.__doc__ + "**2"
    return sq

def power_feat(n):

    def fn(f, n=n):
        def pow(e, n=n):
            return f(e) ** n

        pow.__doc__ = f.__doc__ + "**" + str(n)
        return pow

    return fn

def prod_feat(f1, f2):

    def feat(e):
        return f1(e) * f2(e)

    f1._doc = f1.__doc__ if f1.__doc__ is not None else "f1"
    f2._doc = f2.__doc__ if f2.__doc__ is not None else "f2"
    
    feat.__doc__ = f1._doc + "*" + f2._doc
    return feat

def eq_feat(f1, f2):

    def feat(e):
        return 1 if f1(e) == f2(e) else 0

    f1._doc = f1.__doc__ if f1.__doc__ is not None else "f1"
    f2._doc = f2.__doc__ if f2.__doc__ is not None else "f2"
    
    feat.__doc__ = f1._doc + "==" + f2._doc
    
    return feat

def neq_feat(f1, f2):

    def feat(e):
        return 1 if f1(e) != f2(e) else 0

    f1._doc = f1.__doc__ if f1.__doc__ is not None else "f1"
    f2._doc = f2.__doc__ if f2.__doc__ is not None else "f2"
    
    feat.__doc__ = f1._doc + "!=" + f2._doc
    return feat

class Learner(Displayable):

    def __init__(self, dataset):

        raise NotImplementedError("Learner.__init__")

    def learn(self):

        raise NotImplementedError("learn")
