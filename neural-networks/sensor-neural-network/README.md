# Sistema Inteligente de Detecção de Falhas com Redes Neurais

Projeto de Machine Learning e Deep Learning desenvolvido em Python utilizando TensorFlow e scikit-learn para classificação de estados operacionais de sensores simulados.

O objetivo do projeto é construir um pipeline completo de IA capaz de detectar:

* funcionamento normal;
* estado de alerta;
* falha crítica.

A partir de dados simulados de sensores industriais.

---

# Objetivos do Projeto

Este projeto foi desenvolvido para estudar e aplicar:

* pré-processamento de dados;
* modelos clássicos de Machine Learning;
* redes neurais artificiais;
* classificação multiclasse;
* métricas de avaliação;
* treinamento supervisionado;
* visualização de desempenho;
* organização de pipelines de IA.

Além disso, o projeto foi estruturado pensando em futuras expansões para:

* TinyML;
* ESP32-S3;
* sistemas embarcados inteligentes;
* manutenção preditiva;
* séries temporais.

---

# Tecnologias Utilizadas

* Python
* NumPy
* Pandas
* Matplotlib
* scikit-learn
* TensorFlow

---

# Estrutura do Projeto

```text id="vtt8p6"
sensor-neural-network/
│
├── data/
│   └── sensor_data.csv
│
├── models/
│
├── src/
│   ├── preprocessing.py
│   ├── train_ml.py
│   ├── train_nn.py
│   └── evaluate.py
│
├── requirements.txt
└── README.md
```

---

# Dataset

O dataset é sintético e simula leituras de sensores industriais.

## Features

| Feature      | Descrição               |
| ------------ | ----------------------- |
| temperatura  | Temperatura do sistema  |
| luminosidade | Intensidade luminosa    |
| distancia    | Distância medida        |
| vibracao     | Intensidade de vibração |
| tensao       | Tensão elétrica         |
| corrente     | Corrente elétrica       |

---

# Classes

| Classe | Significado |
| ------ | ----------- |
| 0      | Normal      |
| 1      | Alerta      |
| 2      | Falha       |

---

# Pipeline do Projeto

## 1. Geração do Dataset

Criação de dados sintéticos simulando sensores reais.

## 2. Pré-processamento

* separação treino/teste;
* normalização com StandardScaler;
* preparação das features.

## 3. Modelo Clássico

Treinamento utilizando:

* Random Forest Classifier.

## 4. Rede Neural

Treinamento de uma rede neural densa utilizando:

* camadas Dense;
* ReLU;
* Softmax;
* Adam optimizer.

## 5. Avaliação

Análise de:

* accuracy;
* precision;
* recall;
* F1-score;
* loss;
* curvas de treinamento.

---

# Arquitetura da Rede Neural

```python id="0mhjpv"
model = keras.Sequential([
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])
```

# Principais Aprendizados

Durante o desenvolvimento deste projeto foram estudados:

* Machine Learning supervisionado;
* Redes neurais artificiais;
* normalização de dados;
* overfitting;
* treinamento de modelos;
* separação treino/teste;
* classificação multiclasse;
* otimização usando Adam;
* análise de métricas.

---

# Melhorias Futuras

O projeto poderá evoluir para:

* datasets mais realistas;
* ruído de sensores;
* falhas intermitentes;
* séries temporais;
* LSTM;
* manutenção preditiva;
* exportação para TensorFlow Lite;
* execução em ESP32-S3;
* TinyML;
* inferência em tempo real.

---

# Como Executar

## Instalar dependências

```bash id="d6pcc4"
pip install -r requirements.txt
```

---

## Treinar modelo clássico

```bash id="k1d9ty"
python train_ml.py
```

---

## Treinar rede neural

```bash id="2e2b2n"
python train_nn.py
```

---

# Requisitos

Exemplo de `requirements.txt`:

```text id="tz8nkg"
numpy
pandas
matplotlib
scikit-learn
tensorflow
```

---

# Autor

Projeto desenvolvido para estudos de:

* Machine Learning;
* Deep Learning;
* Inteligência Artificial aplicada;
* Sistemas embarcados inteligentes.
