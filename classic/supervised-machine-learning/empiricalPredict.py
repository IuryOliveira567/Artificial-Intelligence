import pandas as pd
import random
from learnProblem import Evaluate
from learnNoInputs import Predict
from collections import defaultdict

# Carregar dados reais
df = pd.read_csv('data/FF.csv', sep=';')
data = []

# Transforma os números em uma sequência binária, por exemplo: 1 se > média, 0 caso contrário
todos_numeros = []

for row in df.itertuples(index=False):
    nums = list(map(int, row))  # converte a linha em lista de inteiros
    todos_numeros.extend(nums)

media = sum(todos_numeros) / len(todos_numeros)

# Transformar os dados em 0s e 1s com base em se o número é maior que a média
bin_data = [1 if n > media else 0 for n in todos_numeros]

# Função adaptada
def test_no_inputs_from_real_data(data, error_measures=Evaluate.all_criteria, test_size=10, training_sizes=[1, 2, 3, 4, 5, 10, 20, 100, 1000], num_samples=1000):

    for train_size in training_sizes:
        results = {predictor: defaultdict(float) for predictor in Predict._all}

        for _ in range(num_samples):
            if len(data) < train_size + test_size:
                continue

            # Seleciona um ponto aleatório com espaço suficiente para treino + teste
            start = random.randint(0, len(data) - train_size - test_size)
            training = data[start:start + train_size]
            test = data[start + train_size:start + train_size + test_size]

            for predictor in Predict._all:
                prediction = predictor([], training)

                for error_measure in error_measures:
                    results[predictor][error_measure] += sum(
                        error_measure([], prediction, actual) for actual in test
                    ) / test_size

        print(f"\nFor training size {train_size}:")
        print(" Predictor\t", "\t".join(
              [error.__doc__ if error.__doc__ is not None else "" for error in error_measures]))

        for predictor in Predict._all:
            print(f" {predictor.__doc__}",
                  "\t".join("{:.7f}".format(results[predictor][error] / num_samples)
                            for error in error_measures))


if __name__ == "__main__":
   test_no_inputs_from_real_data(bin_data)
