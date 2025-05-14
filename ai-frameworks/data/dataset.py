import pandas as pd
from pandas.plotting import scatter_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import os


class Data_Set():

    def __init__(self, data_path):
        """Initialize the analyzer by loading data from a CSV file."""

        self.data_path, self.filename = os.path.split(data_path)
        self.data = self.load_data()
        self.default_file_format = "png"

    def split_train_test(self, prob_test=0.2, random_state=42):
        """Splits the DataFrame into random train and test subsets."""

        train_set, test_set = train_test_split(self.data, test_size=prob_test,
                                                         random_state=random_state)
        return train_set, test_set
    
    def stratified_split(self, strat_col, bins, labels, test_size=0.2, random_state=42):
        """Performs a stratified split of a DataFrame based on a continuous column."""

        data = self.data.copy()
        data = data.dropna(axis=0, how='any')

        data["__strat_cat__"] = pd.cut(data[strat_col], bins=bins, labels=labels)

        split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

        for train_idx, test_idx in split.split(data, data["__strat_cat__"]):
            self.strat_train_set = data.loc[train_idx].drop(columns=["__strat_cat__"])
            self.strat_test_set = data.loc[test_idx].drop(columns=["__strat_cat__"])
    
        return self.strat_train_set, self.strat_test_set

    def impute_missing_values(self, strategy='mean'):
        """Fill missing values in numerical columns using the specified strategy."""
        
        numeric_cols = self.data.select_dtypes(include='number').columns
        imputer = SimpleImputer(strategy=strategy)
    
        self.data[numeric_cols] = imputer.fit_transform(self.data[numeric_cols])
        print(f"Missing values in numeric columns filled using strategy: '{strategy}'")

    def encode_categoricals(self, drop='first'):
        """Apply OneHotEncoding to categorical columns in the dataset."""

        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns

        if len(categorical_cols) == 0:
           print("No categorical columns to encode.")
           return

        encoder = OneHotEncoder(drop=drop, sparse_output=False)
        encoded_array = encoder.fit_transform(self.data[categorical_cols])

        encoded_df = pd.DataFrame(
            encoded_array, 
            columns=encoder.get_feature_names_out(categorical_cols),
            index=self.data.index
        )

        self.data = pd.concat([self.data.drop(columns=categorical_cols), encoded_df], axis=1)
        print(f"Encoded columns: {list(categorical_cols)}")

    def load_data(self):
        """Load dataset from the specified file path."""

        csv_path = os.path.join(self.data_path, self.filename)
        return pd.read_csv(csv_path)

    def show_null_counts(self):
        """Display the number of missing (null) values per column."""

        null_counts = self.data.isnull().sum()
        print(null_counts[null_counts > 0])

    def plot_null_heatmap(self, filename=None, **kwargs):
        """Plot a heatmap showing the missing values in the dataset."""

        plt.figure(figsize=(12, 8))
        sns.heatmap(self.data.isnull(), **kwargs)
        plt.title("Missing Values Heatmap")
        self.plot(filename)
    
    def plot(self, filename=None):
        """Save the plot to a file if filename is provided and display it."""
        
        if(filename):
            plt.savefig(filename, format=self.default_file_format, dpi=300, bbox_inches="tight")
            
        plt.show()
        
    def describe(self):
        """Return basic descriptive statistics of the dataset."""

        return self.data.describe()

    def info(self):
        """Print a concise summary of the dataset structure."""

        return self.data.info()
    
    def plot_histogram(self, bins=20, fig_size=(15, 20), filename=None, **kwargs):
        """Plot histograms for all numeric features."""

        self.data.hist(bins=bins, figsize=fig_size, **kwargs)
        plt.title("Feature Distributions")
        self.plot(filename)
        
    def plot_correlation_heatmap(self, filename=None, **kwargs):
        """Plot a heatmap showing the correlation matrix of numerical features."""

        numeric_df = self.data.select_dtypes(include='number')
        correlation = numeric_df.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', **kwargs)

        plt.title("Correlations Heatmap")
        self.plot(filename)

    def plot_boxplots(self, x_feature, y_feature, filename=None, **kwargs):
        """Plot a boxplot of a feature grouped by the target variable."""

        sns.boxplot(x=x_feature, y=y_feature, data=self.data, **kwargs)
        plt.title(f"Boxplot of {x_feature} by {y_feature}")
        self.plot(filename)
        
    def plot_pairplot(self, feature, filename=None, **kwargs):
        """Plot pairwise relationships between features, colored by the target."""

        sns.pairplot(self.data, **kwargs)
        plt.title(f"Pairplot of Features Colored by {feature}")
        self.plot(filename)

    def plot_bar(self, feature, filename=None, **kwargs):
        """Plot a bar chart showing the distribution of a categorical variable."""

        sns.countplot(x=feature, data=self.data, **kwargs)
        plt.title(f"Distribution of {feature}")
        self.plot(filename)

    def plot_scatter(self, x_feature, y_feature, filename=None, **kwargs):
        """Plot a scatterplot between two features, optionally colored by a third."""

        sns.scatterplot(x=x_feature, y=y_feature, data=self.data, **kwargs)
        plt.title(f"Scatterplot of {x_feature} vs {y_feature}")
        self.plot(filename)

    def plot_distribution(self, feature, filename=None, **kwargs):
        """Plot the value counts of a target variable as a bar chart."""

        self.data[feature].value_counts().plot(kind='bar', **kwargs)
        plt.title(f"Value Counts of {feature}")
        self.plot(filename)

    def plot_violin(self, x_feature, y_feature, filename=None, **kwargs):
        """Plot a violin plot for a feature relative to the target variable."""

        sns.violinplot(x=x_feature, y=y_feature, data=self.data, **kwargs)
        plt.title(f"Violin Plot of {x_feature} by {y_feature}")
        self.plot(filename)

    def scatter_matrix(self, attributes, fig_size=(12, 8), filename=None, **kwargs):
        """Generates a scatter matrix plot for the specified attributes in the dataset."""

        scatter_matrix(self.data[attributes], figsize=fig_size, **kwargs)
        self.plot(filename)
