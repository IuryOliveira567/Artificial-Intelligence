from learnProblem import Data_set, Data_from_file, Learner, Evaluate
from learnNoInputs import Predict
from learnDT import DT_learner
from learnLinear import sigmoid
import statistics
import random
import matplotlib.pyplot as plt


class Boosted_dataset(Data_set):

    def __init__(self, base_dataset, offset_fun, subsample=1.0):

        self.base_dataset = base_dataset
        self.offset_fun = offset_fun
        self.train = random.sample(base_dataset.train, int(subsample * len(base_dataset.train)))
        self.test = base_dataset.test

        self.input_features = self.base_dataset.input_features

        def newout(e):
            
            return self.base_dataset.target(e) - self.offset_fun(e)

        newout.frange = self.base_dataset.target.frange
        newout.ftype = self.infer_type(newout.frange)
        self.target = newout

    def conditions(self, *args, colsample_bytree=0.5, **nargs):

        conds = self.base_dataset.conditions(*args, **nargs)
        return random.sample(conds, int(colsample_bytree * len(conds)))


class Boosting_learner(Learner):

    def __init__(self, dataset, base_learner_class, subsample=0.8):

        self.dataset = dataset
        self.base_learner_class = base_learner_class
        self.subsample = subsample

        mean = sum(self.dataset.target(e) for e in self.dataset.train) / len(self.dataset.train)
        self.predictor = lambda e: mean
        self.predictor.__doc__ = "lambda e:" + str(mean)
        self.offsets = [self.predictor]
        self.predictors = [self.predictor]
        self.errors = [dataset.evaluate_dataset(dataset.test, self.predictor, Evaluate.squared_loss)]
        self.display(1, "Predict mean test set mean squared loss=", self.errors[0])

    def learn(self, num_ensembles=10):

        for i in range(num_ensembles):
            train_subset = Boosted_dataset(self.dataset, self.predictor, subsample=self.subsample)
            learner = self.base_learner_class(train_subset)
            new_offset = learner.learn()
            self.offsets.append(new_offset)

            def new_pred(e, old_pred=self.predictor, off=new_offset):

                return old_pred(e) + off(e)

            self.predictor = new_pred
            self.predictors.append(new_pred)
            self.errors.append(data.evaluate_dataset(data.test, self.predictor, Evaluate.squared_loss))

            self.display(1, f"Iteration {len(self.offsets) - 1}, treesize={new_offset.num_leaves}. mean\
                                  squared loss={self.errors[-1]}")

            return self.predictor

def sp_DT_learner(split_to_optimize=Evaluate.squared_loss, leaf_prediction=Predict.mean, **nargs):

    def new_learner(dataset):

        return DT_learner(dataset, split_to_optimize=split_to_optimize,
                          leaf_prediction=leaf_prediction, **nargs)

    return new_learner


def plot_boosting_trees(data, steps=10, mcws=[30, 20, 20, 10], gammas=[100, 200, 300, 500]):

    learners = [(mcw, gamma, Boosting_learner(data,
                                        sp_DT_learner(min_child_weight=mcw, gamma=gamma)))
                 for gamma in gammas for mcw in mcws]

    plt.ion()
    plt.xscale('linear')
    plt.xlabel("number of trees")
    plt.ylabel("mean squared loss")

    markers = (m + c for c in ['k', 'g', 'r', 'b', 'm', 'c', 'y'] for m in ['-', '--', '-.', ':'])

    for (mcw, gamma, learner) in learners:
        data.display(1, f"min_child_weight={mcw}, gamma={gamma}")
        learner.learn(steps)

        plt.plot(range(steps + 1), learner.errors, next(markers),
                 labe=f"min_child_weight={mcw}, gamma={gamma}")
        plt.legend()
        plt.draw()


class GTB_learner(DT_learner):

    def __init__(self, dataset, num_trees, lambda_reg=1, gamma=0, **dtargs):

        DT_learner.__init__(self, dataset, split_to_optimize=Evaluate.log_loss, **dtargs)

        self.number_trees = number_trees
        self.lambda_reg = lambda_reg
        self.gamma = gamma
        self.trees = []

    def learn(self):

        for i in range(self.number_trees):
            tree = self.learn_tree(self.dataset.conditions(self.max_num_cuts), self.train)
            self.trees.append(tree)
            self.display(1, f"""Iteration {i} treesize = {tree.num_leaves}
                     trian logloss={self.dataset.evaluate_dataset(self.dataset.train,
                     self.gtb_predictor, Evaluate.log_loss)} test logloss={
                     self.dataset.evaluate_dataset(self.dataset.test,
                     self.gtb_predictor, Evaluate.log_loss)}""")

        return self.gtb_predictor

    def gtb_predictor(self, example, extra=0):

        return sigmoid(sum(t(example) for t in self.trees) + extra)

    def leaf_values(self, egs, domain=[0, 1]):

        pred_acts = [(self.gtb_predictor(e), self.target(e)) for e in egs]

        return sum(a - p for (p, a) in pred_acts) / (sum(p*(1-p) for (p, a) in
                                                         pred_acts) + self.lambda_reg)

    def sum_losses(self, data_subset):

        leaf_val = self.leaf_value(data_subset)
        error = sum(Evaluate.log_loss(self.gtb_predictor(e, leaf_val),
                                      self.target(e))
                    for e in data_subset) + self.gamma

        return error
