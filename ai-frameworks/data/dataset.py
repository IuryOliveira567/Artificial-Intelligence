import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


class Data_Set():

    def __init__(self, data_path):

        self.data_path = "\\".join(data_path.split("\\")[:-1])
        self.filename = data_path.split("\\")[-1]
        self.data = self.load_data()

    def describe(self):

        return self.data.describe
    
    def load_data(self):

        csv_path = os.path.join(self.data_path, self.filename)
        return pd.read_csv(csv_path)

    def plot_histogram(self, bins=20, fig_size=(15, 20)):

        self.data.hist(bins=bins, figsize=fig_size)
        plt.show()

    def plot_correlation_heatmap(self):

        numeric_df = self.data.select_dtypes(include='number')
        correlation = numeric_df.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation, annot=True, cmap='coolwarm')

        plt.title("Correlations Heatmap")
        plt.show()

    def plot_boxplots(self, target, col):

        sns.boxplot(x=target, y=col, data=self.data)
        plt.show()
        
    def plot_pairplot(self, target):

        sns.pairplot(self.data, hue=target)

    def plot_bar(self, target):

        sns.countplot(x=target, data=self.data)
        plt.show()

    def plot_scatter(self, h_line, v_line, hue=False):

        sns.scatterplot(x=h_line, y=v_line, data=self.data)
        plt.show()

    def plot_distribution(self, target):

        self.data[target].value_counts().plot(kind='bar')
        plt.show()

    def plot_violin(self, _class, feature):

        sns.violinplot(x=_class, y=feature, data=self.data)
        plt.show()
