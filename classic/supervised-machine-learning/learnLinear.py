from learnProblem import Learner, Evaluate, Data_from_file, Data_set_augmented, power_feat
import random, math
import matplotlib.pyplot as plt


class Linear_learner(Learner):

    def __init__(self, dataset, train=None, learning_rate=0.1,
                 max_unit=0.2, squashed=True, batch_size=10):

        self.dataset = dataset
        self.target = dataset.target

        if(train == None):
            self.train = self.dataset.train
        else:
            self.train = train

        self.learning_rate = learning_rate
        self.squashed = squashed
        self.batch_size = batch_size
        self.input_features = [one] + dataset.input_features

        self.weights = {feat: random.uniform(-max_unit, max_unit)
                        for feat in self.input_features}
        
    def predictor(self, e):

        linpred = sum(w * f(e) for f, w in self.weights.items())

        if(self.squashed):
            return sigmoid(linpred)
        else:
            return linpred

    def predictor_string(self, sig_dig=3):

        doc = "+".join(str(round(val, sig_dig)) + "*" + str(feat.__doc__)
                       for feat, val in self.weights.items())

        if(self.squashed):
            return "sigmoid(" + doc + ")"
        else:
            return doc

    def learn(self, num_iter=100):

        batch_size = min(self.batch_size, len(self.train))
        d = {feat: 0 for feat in self.weights}

        for it in range(num_iter):
            self.display(2, "prediction=", self.predictor_string())

            for e in random.sample(self.train, batch_size):
                error = self.predictor(e) - self.target(e)
                update = self.learning_rate * error

                for feat in self.weights:
                    d[feat] += update * feat(e)

            for feat in self.weights:
                self.weights[feat] -= d[feat]
                d[feat] = 0

        return self.predictor

def one(e):
    "1"
    return 1

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def logit(x):
    return -math.log(1 / x - 1)

def softmax(xs, domain=None):

    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)

    if(domain):
        return {d: v / s for(d, v) in zip(domain, exps)}
    else:
        return [v / s for v in exps]

def indicator(v, domain):

    return [1 if v == dv else 0 for dv in domain]

def test(**args):
    class_map = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
    data = Data_from_file('data/iris.data', target_index=-1)
    data.target = lambda e: class_map[e[-1]]
    
    learner = Linear_learner(data, squashed=False, learning_rate=0.005, **args)
    learner.learn()
    
    for ecrit in Evaluate.all_criteria:
        test_error = data.evaluate_dataset(data.train, learner.predictor, ecrit)
        print("      Average", ecrit.__doc__, "is", test_error)

def plot_steps(learner=None, data=None, criterion=Evaluate.squared_loss, step=1,
               num_steps=1000, log_scale=True, legend_label=""):

    if(legend_label != ""): legend_label += " "

    plt.ion()
    plt.xlabel("step")
    plt.ylabel("Everage " + criterion.__doc__)

    if(log_scale):
        plt.xscale('log')
    else:
        plt.xscale('linear')

    if(data is None):
        data = Data_from_file('data/Iris.data', has_header=True,
                              num_train=19, target_index=0, one_hot=True)

    if(learner is None):
        learner = Linear_learner(data, data.train, learning_rate=0.005, squashed=False)

    train_errors = []
    test_errors = []

    for i in range(1, num_steps + 1, step):
        test_errors.append(data.evaluate_dataset(data.test, learner.predictor, criterion))
        train_errors.append(data.evaluate_dataset(data.test, learner.predictor, criterion))

        learner.display(2, "Train error: ", train_errors[-1], "Test error: ", test_errors[-1])
        learner.learn(num_iter=step)

    plt.plot(range(1, num_steps + 1, step), train_errors, ls='-', label=legend_label + "training")
    plt.plot(range(1, num_steps + 1, step), test_errors, ls='--', label=legend_label + "test")
    plt.legend()
    plt.draw()

    learner.display(1, "Train error:", train_errors[-1],
                    "Test error:", test_errors[-1])

def arange(start, stop, step):

    while(start < stop):
        yield start
        start += step

def plot_prediction(data, learner=None, minx=0, maxx=5, step_size=0.01, label="function"):

    plt.ion()
    plt.xlabel("x")
    plt.ylabel("y")

    if(learner is None):
        learner = Linear_learner(data, squashed=True)

    learner.learning_rate= 0.001
    learner.learn(100)

    learner.learning_rate= 0.0001
    learner.learn(1000)

    learner.learning_rate= 0.00001
    learner.learn(10000)

    learner.display(1, "function learned is", learner.predictor_string(),
                    "error=", data.evaluate_dataset(data.train, learner.predictor,
                                                    Evaluate.squared_loss))

    plt.plot([e[0] for e in data.train], [e[-1] for e in data.train], "bo", label="data")
    plt.plot(list(arange(minx, maxx, step_size)), [learner.predictor([x])
                                                   for x in arange(minx, maxx, step_size)], label=label)

    plt.legend()
    plt.draw()


def plot_polynomials(data, learner_class=Linear_learner,
                     max_degree=5, minx=0, maxx=5, num_iter=1000000,
                     learning_rate=0.00001, step_size=0.01):

    plt.ion()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot([e[0] for e in data.train], [e[-1] for e in data.train], "ko", label="data")

    x_values = list(arange(minx, maxx, step_size))
    line_style = ['-', '--', '-.', ':']
    colors = ['0.5', 'k', 'k', 'k', 'k']

    for degree in range(max_degree):
        data_aug = Data_set_augmented(data, [power_feat(n) for n in range(1, degree + 1)],
                                      include_orig=False)

        learner = learner_class(data_aug, squashed=False)
        learner.learning_rate = learning_rate
        learner.learn(num_iter)
        learner.display(1, "For degree", degree, "function learned is", learner.predictor_string(),
                        "error", data.evaluate_dataset(data.train, learner.predictor, Evaluate.squared_loss))

        ls = line_style[degree % len(line_style)]
        col = colors[degree % len(colors)]

        plt.plot(x_values, [learner.predictor([x]) for x in x_values],
        linestyle=ls, color=col, label="degree=" + str(degree))
        plt.legend(loc='upper left')
        plt.draw()
