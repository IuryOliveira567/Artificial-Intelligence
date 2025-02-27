from learnProblem import Evaluate
import math, random, collections, statistics
import utilities


class Predict(object):

    def empirical(self, data, domain=[0, 1], icount=0):
        """Empirical prediction."""

        counts = {v: icount for v in domain}

        for e in data:
            counts[e] += 1

        s = sum(counts.values())
        return {k: v / s for (k, v) in counts.items()}

    def bounded_empirical(self, data, domain=[0, 1], bound=0.01):
        """Bounded empirical prediction."""

        return {k: min(max(v, bound), 1 - bound) for(k, v) in
                Predict.empirical(data, domain).items()}

    def laplace(self, data, domain=[0, 1]):
        """Laplace prediction."""

        return Predict.empirical(data, domain, icount=1)

    def cmode(self, data, domain=[0, 1]):
        """Categorical mode prediction."""

        md = statistics.mode(data)
        return {v: 1 if v == md else 0 for v in domain}

    def cmedian(self, data, domain=[0, 1]):
        """Categorical median prediction."""

        md = statistics.median_low(data)
        return {v: 1 if v == md else 0 for v in domain}

    def mean(self, data, domain=[0, 1]):
        """Mean prediction."""

        return statistics.mean(data)

    def rmean(self, data, domain=[0, 1], mean0=0, pseudo_count=1):
        """Weighted mean prediction."""

        sum = mean0 * pseudo_count
        count = pseudo_count
        
        for e in data:
            sum += e
            count += 1

        return sum / count

    def mode(self, data, domain=[0, 1]):
        """Mode prediction."""

        return statistics.mode(data)

    def median(self, data, domain=[0, 1]):
        """Median prediction."""

        return statistics.median(data)

    _all = [empirical, mean, rmean, bounded_empirical, laplace, cmode, mode, median, cmedian]
    select = {"boolean": [empirical, bounded_empirical, laplace, cmode, cmedian],
              "categorical": [empirical, bounded_empirical, laplace, cmode, cmedian],
              "numeric": [mean, rmean, mode, median]}
    
def test_no_inputs(error_measures = Evaluate.all_criteria, num_samples=10000,
                   test_size=10, training_sizes=[1, 2, 3, 4, 5, 10, 20, 100, 1000]):

    for train_size in training_sizes:
        results = {predictor: {error_measure: 0 for error_measure in error_measures}
                   for predictor in Predict._all}

        for sample in range(num_samples):
            prob = random.random()

            training = [1 if random.random() < prob else 0 for i in range(train_size)]
            test = [1 if random.random() < prob else 0 for i in range(test_size)]

            predict = Predict()
            for predictor in predict._all:
                prediction = predictor(predict, training)

                for error_measure in error_measures:
                    results[predictor][error_measure] += sum(error_measure(prediction, actual) for actual
                                                             in test) / test_size

        print(f"For training size {train_size}:")
        print(" Predictor\t", "\t".join(
                  [error_measure.__doc__  if error_measure.__doc__ is not None else ""
                   for error_measure in error_measures]))

        for predictor in Predict._all:
              print(f" {predictor.__doc__}",
                    "\t".join("{:.7f}".format(results[predictor][error_measure] / num_samples)
                              for error_measure in error_measures))
