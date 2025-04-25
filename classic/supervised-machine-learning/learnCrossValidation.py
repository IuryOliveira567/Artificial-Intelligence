from learnProblem import Data_set, Data_from_file, Evaluate
from learnNoInputs import Predict
from learnDT import DT_learner
import matplotlib.pyplot as plt
import random


class K_fold_dataset(object):

    def __init__(self, training_set, num_folds):

        self.data = training_set.train.copy()
        self.target = training_set.target
        self.input_features = training_set.input_features
        self.num_folds = num_folds
        self.conditions = training_set.conditions

        random.shuffle(self.data)
        self.fold_boundaries = [(len(self.data) * i) // num_folds
                                for i in range(0, num_folds + 1)]

    def fold(self, fold_num):

        for i in range(self.fold_boundaries[fold_num],
                       self.fold_boundaries[fold_num + 1]):
            yield self.data[i]

    def fold_complement(self, fold_num):

        for i in range(0, self.fold_boundaries[fold_num]):
            yield self.data[i]

        for i in range(self.fold_boundaries[fold_num + 1], len(self.data)):
            yield self.data[i]


    def validation_error(self, learner, error_measure, **other_params):

        error =  0
        
        try:
            for i in range(self.num_folds):
                predictor = learner(self, train=list(self.fold_complement(i)),
                                    **other_params).learn()
                error += sum(error_measure(predictor(e), self.target(e))
                             for e in self.fold(i))
        except ValueError:
            return float("inf")

        return error / len(self.data)

def plot_error(data, criterion=Evaluate().squared_loss, leaf_prediction=Predict().empirical,
               num_folds=4, maxx=None, xscale='linear'):

    plt.ion()
    plt.xscale(xscale)
    plt.xlabel("min_child_weight")
    plt.ylabel("average " + criterion.__doc__)

    folded_data = K_fold_dataset(data, num_folds)

    if(maxx == None):
        maxx = len(data.train) // 2 + 1

    verrors = []
    terrors = []

    for mcw in range(1, maxx):
        verrors.append(folded_data.validation_error(DT_learner, criterion, leaf_prediction=leaf_prediction,
                                    min_child_weight=mcw))
        tree = DT_learner(data, criterion, leaf_prediction=leaf_prediction,
                          min_child_weight=mcw).learn()
        
        terrors.append(data.evaluate_dataset(data.test, tree, criterion))

    plt.plot(range(1, maxx), verrors, ls='-', color='k', label="validation for " + criterion.__doc__)
    plt.plot(range(1, maxx), terrors, ls='--', color='k', label="test set for " + criterion.__doc__)

    plt.legend("Errors comparison")
    plt.draw()
