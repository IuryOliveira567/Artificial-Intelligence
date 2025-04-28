import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


class Data_Set():

    def __init__(self, data_path):
        """Initialize the analyzer by loading data from a CSV file."""

        self.data_path = "\\".join(data_path.split("\\")[:-1])
        self.filename = data_path.split("\\")[-1]
        self.data = self.load_data()
    
    def load_data(self):
        """Load dataset from the specified file path."""

        csv_path = os.path.join(self.data_path, self.filename)
        return pd.read_csv(csv_path)

    def describe(self):
        """Return basic descriptive statistics of the dataset."""

        return self.data.describe()
        
    def plot_histogram(self, bins=20, fig_size=(15, 20)):
        """Plot histograms for all numeric features."""

        self.data.hist(bins=bins, figsize=fig_size)
        plt.show()

    def plot_correlation_heatmap(self):
        """Plot a heatmap showing the correlation matrix of numerical features."""

        numeric_df = self.data.select_dtypes(include='number')
        correlation = numeric_df.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation, annot=True, cmap='coolwarm')

        plt.title("Correlations Heatmap")
        
        plt.show()

    def plot_boxplots(self, target, col):
        """Plot a boxplot of a feature grouped by the target variable."""

        sns.boxplot(x=target, y=col, data=self.data)
        plt.show()
        
    def plot_pairplot(self, target):
        """Plot pairwise relationships between features, colored by the target."""

        sns.pairplot(self.data, hue=target)
        plt.show()

    def plot_bar(self, target):
        """Plot a bar chart showing the distribution of a categorical variable."""

        sns.countplot(x=target, data=self.data)
        plt.show()

    def plot_scatter(self, h_line, v_line, hue=None):
        """Plot a scatterplot between two features, optionally colored by a third."""

        sns.scatterplot(x=h_line, y=v_line, hue=hue, data=self.data)
        plt.show()

    def plot_distribution(self, target):
        """Plot the value counts of a target variable as a bar chart."""

        self.data[target].value_counts().plot(kind='bar')
        plt.show()

    def plot_violin(self, _class, feature):
        """Plot a violin plot for a feature relative to the target variable."""

        sns.violinplot(x=_class, y=feature, data=self.data)
        plt.show()
