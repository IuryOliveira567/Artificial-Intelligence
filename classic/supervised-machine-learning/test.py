from learnProblem import Data_set, Data_from_file, Data_from_files, Data_set_augmented, prod_feat, neq_feat, Evaluate, square
from learnNoInputs import Predict, test_no_inputs
import random
from learnDT import DT_learner
from learnCrossValidation import plot_error, K_fold_dataset
from learnLinear import Linear_learner, plot_steps, test, plot_prediction, plot_polynomials
from learnBoosting import Boosted_dataset, plot_boosting_trees


if __name__ == "__main__":
   data = Data_from_file('data/iris.data', prob_test=1/3, target_index=-1)
   #dataplus = Data_set_augmented(data,[],[prod_feat])
   #emp = Predict().laplace(data=[1, 1, 2, 3, 1, 3, 5, 6, 7, 1, 2, 5, 8], domain=[1, 2, 3, 4, 5, 6, 7, 8, 9])
   #dset = Data_set(data.train, target_index=0)
   #target_values = [dset.target(e) for e in data.train]
   #p = Predict()
   #ev = dset.evaluate_dataset(dset.train, p.mean, Evaluate().squared_loss, label=-1)
   #print("Acurracy : ", accuracy(ev, 5.4))
   
   #dataplus = Data_set_augmented(data, [], [prod_feat, neq_feat])
   #f = Predict()
   
   #dt = DT_learner(data, max_num_cuts=4, leaf_prediction=Predict().empirical)
   #tree = dt.learn()
   #e = data.train[0]
   #r = tree(e)
   
   #print("-" * 80)
   
   #data2 = Data_from_file('data/iris.data', target_index=0, seed=123)
   #kfold = K_fold_dataset(data, 4)
   #res = kfold.validation_error(DT_learner, Evaluate().log_loss)
   #plot_error(data)
   #error = kfold.validation_error(DT_learner, Evaluate.squared_loss)
   #plot_error(data, criterion=Evaluate.squared_loss, leaf_prediction=Predict().empirical, maxx=4)
   f = Linear_learner(data, data.train, learning_rate=0.005, squashed=False)
   g = f.learn()
   #plot_prediction(data, f)
   #plot_polynomials(data)
   #plot_steps(f, data=data, num_steps=1000)
   #test()
   #plot_boosting_trees(data)

   #data = Data_from_file('data/iris.data', prob_test=0.5, target_index=0, seed=123)
   #dataplus = Data_set_augmented(data, [], [prod_feat])
   #plot_steps(data=dataplus, num_steps=1000)
   #plot_steps(data=dataplus, num_steps=1000) # warning very slow

