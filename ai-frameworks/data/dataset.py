import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


class Data_Set():

    def __init__(self, data_path):
        """Initialize the analyzer by loading data from a CSV file."""

        self.data_path, self.filename = os.path.split(data_path)
        self.data = self.load_data()
        self.default_file_format = "png"
    
    def load_data(self):
        """Load dataset from the specified file path."""

        csv_path = os.path.join(self.data_path, self.filename)
        return pd.read_csv(csv_path)

    def show_null_counts(self):
        """Display the number of missing (null) values per column."""

        null_counts = self.data.isnull().sum()
        print(null_counts[null_counts > 0])

    def plot_null_heatmap(self, filename=None):
        """Plot a heatmap showing the missing values in the dataset."""

        plt.figure(figsize=(12, 8))
        sns.heatmap(self.data.isnull(), cbar=False, cmap='viridis')
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
    
    def plot_histogram(self, bins=20, fig_size=(15, 20), filename=None):
        """Plot histograms for all numeric features."""

        self.data.hist(bins=bins, figsize=fig_size)
        plt.title("Feature Distributions")
        self.plot(filename)

    def plot_correlation_heatmap(self, filename=None):
        """Plot a heatmap showing the correlation matrix of numerical features."""

        numeric_df = self.data.select_dtypes(include='number')
        correlation = numeric_df.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation, annot=True, cmap='coolwarm')

        plt.title("Correlations Heatmap")
        self.plot(filename)

    def plot_boxplots(self, x_feature, y_feature, filename=None):
        """Plot a boxplot of a feature grouped by the target variable."""

        sns.boxplot(x=x_feature, y=y_feature, data=self.data)
        plt.title(f"Boxplot of {x_feature} by {y_feature}")
        self.plot(filename)
        
    def plot_pairplot(self, feature, filename=None):
        """Plot pairwise relationships between features, colored by the target."""

        sns.pairplot(self.data, hue=feature)
        plt.title(f"Pairplot of Features Colored by {feature}")
        self.plot(filename)

    def plot_bar(self, feature, filename=None):
        """Plot a bar chart showing the distribution of a categorical variable."""

        sns.countplot(x=feature, data=self.data)
        plt.title(f"Distribution of {feature}")
        self.plot(filename)

    def plot_scatter(self, x_feature, y_feature, hue=None, filename=None):
        """Plot a scatterplot between two features, optionally colored by a third."""

        sns.scatterplot(x=x_feature, y=y_feature, hue=hue, data=self.data)
        plt.title(f"Scatterplot of {x_feature} vs {y_feature}")
        self.plot(filename)

    def plot_distribution(self, feature, filename=None):
        """Plot the value counts of a target variable as a bar chart."""

        self.data[feature].value_counts().plot(kind='bar')
        plt.title(f"Value Counts of {feature}")
        self.plot(filename)

    def plot_violin(self, x_feature, y_feature, filename=None):
        """Plot a violin plot for a feature relative to the target variable."""

        sns.violinplot(x=x_feature, y=y_feature, data=self.data)
        plt.title(f"Violin Plot of {x_feature} by {y_feature}")
        self.plot(filename)
