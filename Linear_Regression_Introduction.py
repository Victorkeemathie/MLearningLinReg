# Introduction to Machine Learning

# The main task of Machine Learning is to to explore and construct algorithms that can learn from historical data and make predictions on new input data
# Machine Learning can be broadly classified into :

# 1. Unsupervised Learning - when data contains indicative signals without any description attached, it is up to us to find structure of the data underneath, to discover hidden information or to determine how to describe the data
# This kind of learning data is called unlabelled data 
# Unsupervised learning can be used to detect anomalies such as fraud or defective equipment, or to group customers with similar online behaviours for a mareting campaigns

# 2. Supervised Learning - when learning data comes with description, targets or desired outputs besides indicative signals, the learning goal becomes to find a general rule that maps inputs to outputs
# This kind of learninf data is called labelled data. The learned rule is then used to label new data with unknown outputs. The labels are usually provided by event logging systems and human experts
# Besides, if it is feasible, they may be also reproduced by members of the public through crowdsourcing for instance.
# Supervised Learning is commonly used in daily applications such as face and speech recognition, products or movie recommendations, and sales forecasting
# We can further subdivide supervised machine learning into:-
# Regression which trains on and predicts a continous-valued response for example predicting house prices
# Classification that attempts to find the appropriate class label such as analyzing positive / negative sentiment and prediction loan defaults

# 3. Reinforcement Learning - learning data provides feedbac so that the system adapts to dynamic conditions in order to achieve certain goals
# The system evaluates its performance based on the feedback resposnses and accordingly
# The best known instances include self driving cars and chess master AlphaGo

 
 # Overfitting, Underfitting and the Bias-Variance Tradeoff
 
 # 1. Overfitting
 # This simply means that we are over extracting too much information from the training sets and making our model just work well with them, which is called low bias in machine learning
 # However, at the same time, it will not help us generalize with data and derive patterns from them
 # The model as a result will perform poorly on datasets that were not seen before. This is called high variance in machine learning
 # Overfitting occurs when we try to describe the learning rules based on a relatively small number of observation, instead of the underlying relationship
 # Overfitting takes place when we make the model excessively complex so that it fiits every training sample 
 
 # 2. Underfitting 
 # When a model is underfit, it does not perform well on the training sets, and will not so on the testing sets which means it fails to capture the underlying trend of data
 # Underfitting may occur if we are not using enough data to train the model, or if we are trying to fit a wrong model to the data
 # In Machine Learning, we call Underfitting a situation of high bias and low variance
 
 # Bias - this is the error stemming from incorrect assumptions in the learning algorithm
 
 # Avoiding Overfitting
 # We can avoid overfitting using techniques such as:
 
 # a. Cross-Validation
 # In one round of cross-validation, the original data is divided into two subsets, for training and testing (or validation)
 # The testing performance is recorded
 # Similarly, multiple rounds of cross-validation are performed under different partitions
 # Testing results are then averaged to give a more accurate estimate of model prediction performance 
 # Cross-Validation helps reduce variability and therefore can limit problems like overfitting
 # There are mainly two cross-validations schemes in use: - Exhaustive and - Non-exhaustive
 
 # b. Regularization
 # Unnecessary complexity of the model is a source of overfitting 
 # Regularizations adds extra paramters to the error function we are trying to minimize in order to to penalize complex models
 # According to Occam's razor, simpler methods are to be favoured
 # We can use regularization to reduce influence of the high orders of polynomial by imposing penalties on them
 # This will discourage complexity, even though a less accurate and less strict rule is learned from the training data
 
 # The Bias-Variance Trade-off
 # The prediction of any Machine Learning Algorithm can be broken into: Bias Error, Variance Error, and Irreducible Error
 # The irreducible error cannot be reduced regardless of what algorithm is used
 # It is the error introduced from the chosen framing of the problem and may be caused by factors like unknown variables that influence the mapping of the input variables to the output variable.
 
 # Bias Error 
 # Bias are the simplifying assumptions made by a model to make the target easier to learn
 # Low Bias - Suggests more assumptions about the form of the target function
 # High Bias - Suggests less assumptions about the form of the target function
 
 # Variance Error
 # Variance is the amount that the estimate of the target function will change if different training data is used
 # Low Variance - Suggests small changes to the estimate of the target function with changes to the training dataset
 # High Variance - Suggests large changes to the estimate of the target function with changes to the training data set

# Bias Variance Trade off
# The goal of any supervised machine learning algorithm is to achieve low bias and low variance 
# Note that  Increasing the bias will decrease the variance and Increasing the variance will decrease the bias
# There is a trade-off at play between these two concerns and the algorithms you choose and the way you choose to configure them are finding different balances in this trade-off for your problem

# Linear Regression
# Linear Regression is a statistical method used to model the relationship between a dependent variable (also known as the target or outcome variable) and one or more independent variables (also known as predictors or features). 
# It assumes a linear relationship between the variables, meaning that the change in the dependent variable is directly proportional to the change in the independent 

# 1. Machine Learning:

# Predicting the price of a used car based on features like mileage, age, and brand.
# Estimating the salary of an employee based on their years of experience, education level, and position.
# Predicting the sales of a product based on advertising expenditure and other marketing variables.

# 2. Econometrics:

# Examining the relationship between unemployment rates and GDP growth to understand the impact of economic changes.
# Analyzing the relationship between inflation and interest rates to study monetary policies.
# Estimating the demand for a product based on its price and income of consumers.

# 3. Actuarial Science:

# Modeling the relationship between age, gender, and health factors to predict life expectancy for insurance purposes.
# Analyzing the impact of various risk factors on insurance claims, such as weather conditions for property insurance.
# Estimating the mortality rate of a population based on historical data and demographic factors.
# Weather Conditions: Insurance companies often want to understand how weather conditions affect property damage and claims. Linear Regression can help model the relationship between weather variables (e.g., temperature, precipitation, wind speed) and the frequency or severity of claims for different types of properties.
# Location Factors: Geographic location plays a significant role in property insurance claims. Linear Regression can be employed to analyze how factors like proximity to coastlines, floodplains, or high-crime areas impact the likelihood and cost of insurance claims for properties in different regions.
# Building Characteristics: The features of a building, such as its age, size, construction materials, and safety measures, can influence the severity of claims. Linear Regression can help insurers understand how these building-specific variables correlate with claim amounts.
# Policy Coverage: The terms of insurance policies, including deductibles, coverage limits, and policy add-ons, can affect the frequency and severity of claims. Linear Regression can be utilized to explore the relationship between policy coverage variables and claims data.
# Claim History: Analyzing historical claims data can reveal patterns and insights. Linear Regression can be applied to assess the impact of previous claims on the likelihood of future claims for a particular property or policyholder.
# Demographic Factors: In some cases, demographic factors such as the age, occupation, or lifestyle of property owners may play a role in insurance claims. Linear Regression can help identify correlations between these demographic variables and claims patterns.
# Environmental and Climate Changes: As climate change impacts weather patterns, it can also affect the frequency and severity of property damage. Linear Regression can assist in studying how environmental and climate changes relate to insurance claims over time

# 4. Social Sciences:

# Studying the relationship between education level and income to analyze socioeconomic disparities.
# Analyzing the impact of social media usage on individuals' mental well-being.
# Predicting crime rates based on demographic and economic variables.

# 5. Natural Sciences:

# Modeling the relationship between temperature and the rate of a chemical reaction.
# Estimating the growth of a population of organisms based on environmental factors.
# Analyzing the relationship between rainfall and vegetation growth.

# 6. Business and Marketing:

# Predicting customer churn based on customer behavior and engagement with a product or service.
# Estimating the demand for a new product based on consumer preferences and market trends.
# Analyzing the impact of pricing strategies on sales and revenue.

# 7. Healthcare:

# Modeling the relationship between dosage and treatment efficacy in clinical trials.
# Predicting patient readmission rates based on health history and medical interventions.
# Analyzing the association between lifestyle factors (e.g., exercise, diet) and health outcomes

# Types of Linear Regression:

# 1. Simple Linear Regression: Simple Linear Regression is the most basic form of Linear Regression, involving a single independent variable (predictor) and a single dependent variable (outcome). 
# The relationship between the variables is assumed to be linear, meaning the data points approximately lie on a straight line. 
# The goal of Simple Linear Regression is to fit a line that best represents the relationship between the variables, minimizing the sum of squared differences between the predicted and actual values.

# 2. Multiple Linear Regression: Multiple Linear Regression extends the concept of Simple Linear Regression to involve two or more independent variables. 
# It's used when the dependent variable is influenced by multiple predictors. 
# Each independent variable has its coefficient (slope), representing its impact on the dependent variable, while the intercept represents the predicted value when all independent variables are zero

# 3. Polynomial Regression: Polynomial Regression is a variation of Linear Regression that allows for a nonlinear relationship between the independent and dependent variables. 
# Instead of fitting a straight line, it fits a higher-degree polynomial curve to the data points. This enables us to capture more complex patterns and interactions between the variable

# The best-fit line, also known as the regression line, is the straight line (in the case of Simple Linear Regression) or the hyperplane (in the case of Multiple Linear Regression) that best represents the relationship between the independent variable(s) and the dependent variable. 
# It is the line that minimizes the sum of squared differences between the predicted values and the actual values of the dependent variable based on the given data points



# Machine Learning
# We first start by importing necessary libraries i.e:
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing 
# In scikit-learn, the "datasets" module contains various datasets that are commonly used for practicing and experimenting with machine learning algorithms. 
# These datasets are often used for learning, testing, and demonstrating machine learning techniques in a controlled and reproducible environment
#  "fetch_california_housing" is a specific function within the "datasets" module that allows you to download and access the California housing dataset. 
# This dataset contains data related to housing districts in California, and it is often used for regression tasks, including linear regression

from sklearn.model_selection import train_test_split
# In scikit-learn, the "model_selection" module contains functions and classes related to model selection and evaluation. 
# This module provides tools for splitting data into training and testing sets, cross-validation, and hyperparameter tuning.
# train_test_split: "train_test_split" is a specific function within the "model_selection" module. 
# It is commonly used to split a given dataset into two subsets: the training set and the testing set. 
# This is an essential step in machine learning to evaluate how well a model generalizes to unseen data

from sklearn.linear_model import LinearRegression
# In scikit-learn, the "linear_model" module contains various linear models, including Linear Regression, Lasso Regression, Ridge Regression, and more. 
# Linear Regression is a simple and widely used algorithm for modeling the relationship between a dependent variable (target) and one or more independent variables (features) in a linear manner.
# LinearRegression: "LinearRegression" is a specific class within the "linear_model" module, representing the implementation of the Linear Regression algorithm. 
# When you import this class, you can create an instance of the Linear Regression model to fit a linear relationship between the features and the target variable in your dataset

from sklearn.metrics import mean_absolute_error, mean_squared_error
# metrics: In scikit-learn, the "metrics" module contains various functions and classes to measure the performance of machine learning models. 
# These metrics help evaluate how well a model performs on test data and how accurate its predictions are compared to the true target values.
# mean_absolute_error: "mean_absolute_error" is a specific function within the "metrics" module that calculates the mean absolute error (MAE) between the predicted target values and the true target values. 
# MAE is a metric used in regression tasks to measure the average absolute difference between the predicted and actual values. 
# It provides a measure of the model's accuracy, where lower MAE values indicate better performance.
# mean_squared_error: "mean_squared_error" is another function within the "metrics" module that calculates the mean squared error (MSE) between the predicted target values and the true target values. 
# MSE is a commonly used metric for regression tasks, and it measures the average squared difference between the predicted and actual values. Like MAE, lower MSE values indicate better model performance

# Data Analysis

from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt

# Load the California housing dataset
california = fetch_california_housing()

# Extract features (X) and target variable (y)
X = california.data
y = california.target

# Optional: View feature names and target name
feature_names = california.feature_names
target_name = california.target_names

# Optional: View dataset description
data_description = california.DESCR

# Display dataset description
print("\n=== The Data Description ===")
print(data_description)

# Display target name and feature names
print("\n=== Target Name ===")
print(target_name)
print("\n=== Feature Names ===")
print(feature_names)

# Convert data to a DataFrame
df = pd.DataFrame(X, columns=feature_names)
df['Target (MedHouseValue)'] = y

# Display the shape of the data
print("\n=== Shape of the Data ===")
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# Display the first few rows of the data
print("\n=== First Few Rows of the Data ===")
print(df.head())

# Display the last few rows of the data
print("\n=== Last Few Rows of the Data ===")
print(df.tail())

# Display a random sample of the data
print("\n=== Random Sample of the Data ===")
print(df.sample(5))

# Display summary statistics of the data
print("\n=== Summary Statistics of the Data ===")
print(df.describe())

# Display information about the DataFrame (data types, non-null values, memory usage, etc.)
print("\n=== DataFrame Information ===")
print(df.info())

# Check for missing values
print("\n=== Missing Values ===")
print(df.isnull().sum())

# Display correlation between features
print("\n=== Correlation Between Features ===")
correlation_matrix = df.corr()
print(correlation_matrix)

# Plot histograms of features
print("\n=== Histograms of Features ===")
df.hist(bins=30, figsize=(15, 10))
plt.show()

# Plot scatter plots of features against the target variable
print("\n=== Scatter Plots of Features against Target Variable ===")
for feature in feature_names:
    plt.scatter(X[:, feature_names.index(feature)], y)
    plt.xlabel(feature)
    plt.ylabel('MedHouseValue')
    plt.show()

# Display the target variable distribution
print("\n=== Target Variable Distribution ===")
plt.hist(y, bins=30)
plt.xlabel('MedHouseValue')
plt.ylabel('Frequency')
plt.show()


# EXPLANATORY DATA ANALYSIS

# 1. Pairplot
# Pairplot is a powerful visualization to quickly explore pairwise relationships between numerical features. 
# It creates scatter plots for all combinations of numerical features
# Interpretation: The pairplot displays scatter plots of all numerical features against each other. 
# It helps identify potential linear or non-linear relationships between features. 
# Positive correlations show an upward trend, negative correlations show a downward trend, and no correlation appears as a scattered pattern. 
# The diagonal plots represent the distribution of each feature
# Import necessary libraries
import seaborn as sns   # Import Seaborn for data visualization
import matplotlib.pyplot as plt   # Import Matplotlib for plot rendering
from sklearn.datasets import fetch_california_housing   # Import the California housing dataset from Scikit-learn
import pandas as pd   # Import Pandas for data manipulation and analysis
# Load the California housing dataset
california = fetch_california_housing()   # Fetch the dataset using Scikit-learn
df = pd.DataFrame(california.data, columns=california.feature_names)   # Create a Pandas DataFrame with feature data
df['Target (MedHouseValue)'] = california.target   # Add the target variable (median house value) to the DataFrame
# Create a pairplot of numerical features
sns.pairplot(df)   # Generate scatter plots and histograms for all numerical features against each other
plt.show()   # Display the plot



# 2. Jointplot
# Jointplot displays the joint distribution between two numerical features using a scatter plot with marginal histograms.
# Jointplot between two numerical features
# Interpretation: The jointplot combines a scatter plot with histograms for two numerical features. 
# It helps visualize the bivariate distribution and the individual univariate distributions. 
# The scatter plot shows the relationship between the features, while the histograms show their individual distributions.
# Import necessary libraries
import seaborn as sns   # Import Seaborn for data visualization
import matplotlib.pyplot as plt   # Import Matplotlib for plot rendering
from sklearn.datasets import fetch_california_housing   # Import the California housing dataset from Scikit-learn
import pandas as pd   # Import Pandas for data manipulation and analysis
# Load the California housing dataset
california = fetch_california_housing()   # Fetch the dataset using Scikit-learn
df = pd.DataFrame(california.data, columns=california.feature_names)   # Create a Pandas DataFrame with feature data
df['Target (MedHouseValue)'] = california.target   # Add the target variable (median house value) to the DataFrame
# Create a jointplot between 'MedInc' (Median Income) and 'Target (MedHouseValue)'
sns.jointplot(x='MedInc', y='Target (MedHouseValue)', data=df)   # Generate a scatter plot with histograms for the specified numerical features
plt.show()   # Display the plot

# 3. Boxplot
# Boxplot visualizes the distribution and identifies outliers in a numerical feature
# Boxplot of a numerical feature
# Boxplot of a numerical feature:
# Interpretation: The boxplot provides a summary of the distribution of a numerical feature. 
# The box represents the interquartile range (IQR), with the median line inside it. 
# The whiskers extend to the minimum and maximum values within a defined range (usually 1.5 times the IQR). 
# Outliers beyond the whiskers are shown as individual points. The boxplot helps identify skewness, spread, and potential outliers
# Import necessary libraries
import seaborn as sns   # Import Seaborn for data visualization
import matplotlib.pyplot as plt   # Import Matplotlib for plot rendering
from sklearn.datasets import fetch_california_housing   # Import the California housing dataset from Scikit-learn
import pandas as pd   # Import Pandas for data manipulation and analysis
# Load the California housing dataset
california = fetch_california_housing()   # Fetch the dataset using Scikit-learn
df = pd.DataFrame(california.data, columns=california.feature_names)   # Create a Pandas DataFrame with feature data
df['Target (MedHouseValue)'] = california.target   # Add the target variable (median house value) to the DataFrame
# Create a boxplot of 'MedInc' (Median Income)
sns.boxplot(x='MedInc', data=df)   # Generate a box plot for the specified numerical feature
plt.show()   # Display the plot


# 4. Violin Plot
# Violin plot combines a boxplot with a kernel density plot to show the data's distribution more clearly
# Violin plot of a numerical feature
# Interpretation: The violin plot combines a boxplot with a kernel density plot. 
# The width of the violin at different points represents the density of data, showing where the majority of data points lie. 
# It helps visualize data distribution and density more efficiently than a boxplot
# Import necessary libraries
import seaborn as sns   # Import Seaborn for data visualization
import matplotlib.pyplot as plt   # Import Matplotlib for plot rendering
from sklearn.datasets import fetch_california_housing   # Import the California housing dataset from Scikit-learn
import pandas as pd   # Import Pandas for data manipulation and analysis
# Load the California housing dataset
california = fetch_california_housing()   # Fetch the dataset using Scikit-learn
df = pd.DataFrame(california.data, columns=california.feature_names)   # Create a Pandas DataFrame with feature data
df['Target (MedHouseValue)'] = california.target   # Add the target variable (median house value) to the DataFrame
# Create a violin plot of 'MedInc' (Median Income)
sns.violinplot(x='MedInc', data=df)   # Generate a violin plot for the specified numerical feature
plt.show()   # Display the plot


# 5. Bar plot of a categorical feature:
# Interpretation: The bar plot shows the mean or any other aggregated statistic of a numerical target variable against different categories of a categorical feature. 
# It helps compare the central tendency of the target variable for each category. 
# Higher bars indicate higher means.
# Import necessary libraries
import seaborn as sns   # Import Seaborn for data visualization
import matplotlib.pyplot as plt   # Import Matplotlib for plot rendering
from sklearn.datasets import fetch_california_housing   # Import the California housing dataset from Scikit-learn
import pandas as pd   # Import Pandas for data manipulation and analysis
# Load the California housing dataset
california = fetch_california_housing()   # Fetch the dataset using Scikit-learn
df = pd.DataFrame(california.data, columns=california.feature_names)   # Create a Pandas DataFrame with feature data
df['Target (MedHouseValue)'] = california.target   # Add the target variable (median house value) to the DataFrame
# Create a boxplot of 'MedInc' (Median Income)
sns.boxplot(x='MedInc', data=df)   # Generate a box plot for the specified numerical feature
plt.show()   # Display the plot


# 6. Count plot of a categorical feature:
# Interpretation: The count plot displays the frequency of each category in a categorical feature. 
# It provides a quick overview of the distribution of the categorical variable. 
# The height of each bar represents the count of occurrences
# Import necessary libraries
# Import necessary libraries
import seaborn as sns   # Import Seaborn for data visualization
import matplotlib.pyplot as plt   # Import Matplotlib for plot rendering
from sklearn.datasets import fetch_california_housing   # Import the California housing dataset from Scikit-learn
import pandas as pd   # Import Pandas for data manipulation and analysis
# Load the California housing dataset
california = fetch_california_housing()   # Fetch the dataset using Scikit-learn
df = pd.DataFrame(california.data, columns=california.feature_names)   # Create a Pandas DataFrame with feature data
df['Target (MedHouseValue)'] = california.target   # Add the target variable (median house value) to the DataFrame
# Create a count plot of 'MedInc' (Median Income)
sns.countplot(x='MedInc', data=df)   # Generate a count plot for the specified numerical feature 'MedInc'
plt.show()   # Display the plot


# 7. Distribution plot of a numerical feature:
# Interpretation: The distribution plot (histplot and kdeplot) shows the distribution of a numerical feature. 
# The histogram represents the frequency of values in different bins, and the KDE plot estimates the probability density function. Skewed distributions may be left-skewed (negative skewness) or right-skewed (positive skewness)
# Import necessary libraries
import seaborn as sns   # Import Seaborn for data visualization
import matplotlib.pyplot as plt   # Import Matplotlib for plot rendering
from sklearn.datasets import fetch_california_housing   # Import the California housing dataset from Scikit-learn
import pandas as pd   # Import Pandas for data manipulation and analysis
# Load the California housing dataset
california = fetch_california_housing()   # Fetch the dataset using Scikit-learn
df = pd.DataFrame(california.data, columns=california.feature_names)   # Create a Pandas DataFrame with feature data
df['Target (MedHouseValue)'] = california.target   # Add the target variable (median house value) to the DataFrame
# Create a distribution plot of 'MedInc' (Median Income)
sns.distplot(df['MedInc'])   # Generate a distribution plot for the 'MedInc' feature
plt.show()   # Display the plot


# 8. Categorical boxplot:
# Interpretation: The categorical boxplot displays the distribution of a numerical feature grouped by different categories of a categorical feature. 
# It helps compare the distribution of the numerical feature among different groups. 
# Differences in box heights and whiskers suggest variations between the groups
# Import necessary libraries
# Import necessary libraries
import seaborn as sns   # Import Seaborn for data visualization
import matplotlib.pyplot as plt   # Import Matplotlib for plot rendering
from sklearn.datasets import fetch_california_housing   # Import the California housing dataset from Scikit-learn
import pandas as pd   # Import Pandas for data manipulation and analysis
# Load the California housing dataset
california = fetch_california_housing()   # Fetch the dataset using Scikit-learn
df = pd.DataFrame(california.data, columns=california.feature_names)   # Create a Pandas DataFrame with feature data
df['Target (MedHouseValue)'] = california.target   # Add the target variable (median house value) to the DataFrame
# Use 'MedInc' as categorical data to create the box plot
sns.boxplot(x=pd.cut(df['MedInc'], bins=5), y='Target (MedHouseValue)', data=df)
plt.show()


# 9. Swarm plot of a numerical feature
# Interpretation: The swarm plot displays individual data points of a numerical feature along a categorical axis. 
# It helps visualize the distribution and density of data points. Points closer together indicate higher density
# Import necessary libraries
import seaborn as sns   # Import Seaborn for data visualization
import matplotlib.pyplot as plt   # Import Matplotlib for plot rendering
from sklearn.datasets import fetch_california_housing   # Import the California housing dataset from Scikit-learn
import pandas as pd   # Import Pandas for data manipulation and analysis
# Load the California housing dataset
california = fetch_california_housing()   # Fetch the dataset using Scikit-learn
df = pd.DataFrame(california.data, columns=california.feature_names)   # Create a Pandas DataFrame with feature data
df['Target (MedHouseValue)'] = california.target   # Add the target variable (median house value) to the DataFrame
# Create a swarm plot of 'MedInc' (Median Income)
sns.swarmplot(x=pd.cut(df['MedInc'], bins=5), y='MedInc', data=df)   # Generate a swarm plot to visualize the distribution of 'MedInc' across different bins
plt.show()   # Display the plot


# 10. Parallel coordinates plot:
# Interpretation: Parallel coordinates plot visualizes multivariate data by representing each data point as a line and showing how it interacts with different features. 
# It helps identify patterns and relationships between features. 
# Lines that run closely together may indicate similarity between data points
# Import necessary libraries
import seaborn as sns   # Import Seaborn for data visualization
import matplotlib.pyplot as plt   # Import Matplotlib for plot rendering
from sklearn.datasets import fetch_california_housing   # Import the California housing dataset from Scikit-learn
import pandas as pd   # Import Pandas for data manipulation and analysis
# Load the California housing dataset
california = fetch_california_housing()   # Fetch the dataset using Scikit-learn
df = pd.DataFrame(california.data, columns=california.feature_names)   # Create a Pandas DataFrame with feature data
df['Target (MedHouseValue)'] = california.target   # Add the target variable (median house value) to the DataFrame
# Import parallel_coordinates from pandas.plotting
from pandas.plotting import parallel_coordinates
# Create a parallel coordinates plot based on 'OceanProximity'
parallel_coordinates(df, 'OceanProximity')   # Generate a parallel coordinates plot to visualize the relationship between features across different categories in 'OceanProximity'
plt.show()   # Display the plot


# 11. Andrews curves plot:
# Interpretation: Andrews curves transform each data row into a curve, allowing visualization of multivariate data. 
# It helps observe clusters or patterns in the data. 
# Curves that overlap or follow similar paths suggest similarity between data points
# Import necessary libraries
import seaborn as sns   # Import Seaborn for data visualization
import matplotlib.pyplot as plt   # Import Matplotlib for plot rendering
from sklearn.datasets import fetch_california_housing   # Import the California housing dataset from Scikit-learn
import pandas as pd   # Import Pandas for data manipulation and analysis
# Load the California housing dataset
california = fetch_california_housing()   # Fetch the dataset using Scikit-learn
df = pd.DataFrame(california.data, columns=california.feature_names)   # Create a Pandas DataFrame with feature data
df['Target (MedHouseValue)'] = california.target   # Add the target variable (median house value) to the DataFrame
# Import andrews_curves from pandas.plotting
from pandas.plotting import andrews_curves
# Create an Andrews curves plot using the first three features
andrews_curves(df[['MedInc', 'HouseAge', 'AveRooms', 'Target (MedHouseValue)']], 'Target (MedHouseValue)')
plt.show()   # Display the plot

# 12. PairGrid for customizing pairplots:
# Interpretation: PairGrid allows customization of pairplots with different plot types for upper, diagonal, and lower triangles. 
# It is useful when you want to use different plot types for different parts of the pairplot
# Import necessary libraries
import seaborn as sns   # Import Seaborn for data visualization
import matplotlib.pyplot as plt   # Import Matplotlib for plot rendering
from sklearn.datasets import fetch_california_housing   # Import the California housing dataset from Scikit-learn
import pandas as pd   # Import Pandas for data manipulation and analysis

# Load the California housing dataset
california = fetch_california_housing()   # Fetch the dataset using Scikit-learn
df = pd.DataFrame(california.data, columns=california.feature_names)   # Create a Pandas DataFrame with feature data
df['Target (MedHouseValue)'] = california.target   # Add the target variable (median house value) to the DataFrame

# Create a PairGrid for customizing pairplots
g = sns.PairGrid(df)   # Initialize a PairGrid with the DataFrame 'df' for customizing pairplots

# Use scatterplot for the upper triangle of the PairGrid
g.map_upper(sns.scatterplot)   # Use a scatter plot to visualize the relationship between numerical features in the upper triangle of the PairGrid

# Use histplot for the diagonal of the PairGrid
g.map_diag(sns.histplot)   # Use a histogram to visualize the distribution of each numerical feature along the diagonal of the PairGrid

# Use kdeplot for the lower triangle of the PairGrid
g.map_lower(sns.kdeplot)   # Use a kernel density plot to visualize the joint distribution between numerical features in the lower triangle of the PairGrid

plt.show()   # Display the customized pairplots created using the PairGrid




# 13. Clustermap to visualize hierarchical clustering:
# Interpretation: Clustermap creates a hierarchical cluster heatmap to identify patterns in the data. 
# It helps group similar data points based on their features. 
# Dendrograms show clustering relationships
# Import necessary libraries
import seaborn as sns   # Import Seaborn for data visualization
import matplotlib.pyplot as plt   # Import Matplotlib for plot rendering
from sklearn.datasets import fetch_california_housing   # Import the California housing dataset from Scikit-learn
import pandas as pd   # Import Pandas for data manipulation and analysis

# Load the California housing dataset
california = fetch_california_housing()   # Fetch the dataset using Scikit-learn
df = pd.DataFrame(california.data, columns=california.feature_names)   # Create a Pandas DataFrame with feature data
df['Target (MedHouseValue)'] = california.target   # Add the target variable (median house value) to the DataFrame

# Calculate the correlation matrix
correlation_matrix = df.corr()   # Compute the correlation between numerical features in the DataFrame using the `.corr()` method

# Create a clustermap to visualize hierarchical clustering of the correlation matrix
sns.clustermap(correlation_matrix, cmap='coolwarm')   # Generate a clustered heatmap to show the patterns of correlation between numerical features
plt.show()   # Display the clustered heatmap




# 14. FacetGrid for creating subplots based on categorical variables:
# Interpretation: FacetGrid creates multiple subplots, each representing a subset of the data based on categorical variables. 
# It helps visualize relationships between features across different categories
# Import necessary libraries
import seaborn as sns   # Import Seaborn for data visualization
import matplotlib.pyplot as plt   # Import Matplotlib for plot rendering
from sklearn.datasets import fetch_california_housing   # Import the California housing dataset from Scikit-learn
import pandas as pd   # Import Pandas for data manipulation and analysis

# Load the California housing dataset
california = fetch_california_housing()   # Fetch the dataset using Scikit-learn
df = pd.DataFrame(california.data, columns=california.feature_names)   # Create a Pandas DataFrame with feature data
df['Target (MedHouseValue)'] = california.target   # Add the target variable (median house value) to the DataFrame

# Create a FacetGrid for creating subplots based on categorical variables 'OceanProximity'
g = sns.FacetGrid(df, col='OceanProximity', col_wrap=3)   # Initialize a FacetGrid with 'OceanProximity' as the categorical variable, col_wrap=3 sets the number of columns for the subplots to 3

# Use histplot to create subplots for each 'OceanProximity' category, plotting 'MedInc' (Median Income)
g.map(sns.histplot, 'MedInc')   # Use a histogram for each subplot to visualize the distribution of 'MedInc' (Median Income) for each 'OceanProximity' category

plt.show()   # Display the subplots created using the FacetGrid


# 15. Kernel Density Estimation (KDE) plot:
# Interpretation: The KDE plot estimates the probability density function of a numerical feature. 
# It provides a smooth representation of the data's underlying distribution. 
# Peaks in the KDE suggest modes in the data
# Import necessary libraries
import seaborn as sns   # Import Seaborn for data visualization
import matplotlib.pyplot as plt   # Import Matplotlib for plot rendering
from sklearn.datasets import fetch_california_housing   # Import the California housing dataset from Scikit-learn
import pandas as pd   # Import Pandas for data manipulation and analysis

# Load the California housing dataset
california = fetch_california_housing()   # Fetch the dataset using Scikit-learn
df = pd.DataFrame(california.data, columns=california.feature_names)   # Create a Pandas DataFrame with feature data
df['Target (MedHouseValue)'] = california.target   # Add the target variable (median house value) to the DataFrame

# Create a Kernel Density Estimation (KDE) plot for the 'MedInc' (Median Income) feature
sns.kdeplot(df['MedInc'], shade=True)   # Use a KDE plot to visualize the distribution of 'MedInc' (Median Income), with the area under the curve shaded
plt.show()   # Display the KDE plot



# 16. Residual plot for evaluating linear regression model:
# Interpretation: A residual plot shows the difference between predicted and actual values in a linear regression model. 
# Randomly scattered residuals around zero indicate a good fit, while patterns or trends suggest model inadequacy
import seaborn as sns   # Import Seaborn for data visualization
import matplotlib.pyplot as plt   # Import Matplotlib for plot rendering
from sklearn.datasets import fetch_california_housing   # Import the California housing dataset from Scikit-learn
import pandas as pd   # Import Pandas for data manipulation and analysis

# Load the California housing dataset
california = fetch_california_housing()   # Fetch the dataset using Scikit-learn
df = pd.DataFrame(california.data, columns=california.feature_names)   # Create a Pandas DataFrame with feature data
df['Target (MedHouseValue)'] = california.target   # Add the target variable (median house value) to the DataFrame

# Create a residual plot to evaluate the linear regression model
sns.residplot(x='MedInc', y='Target (MedHouseValue)', data=df)   # Use a residual plot to visualize the residuals (differences between actual and predicted values) for the linear regression model
plt.show()   # Display the residual plot


# 17. Distribution plot of the target variable:
# Interpretation: The distribution plot (histplot or kdeplot) shows the distribution of the target variable. 
# It helps understand its spread and shape. Skewed target distributions may need transformation for modeling
# Import necessary libraries
import seaborn as sns   # Import Seaborn for data visualization
import matplotlib.pyplot as plt   # Import Matplotlib for plot rendering
from sklearn.datasets import fetch_california_housing   # Import the California housing dataset from Scikit-learn
import pandas as pd   # Import Pandas for data manipulation and analysis

# Load the California housing dataset
california = fetch_california_housing()   # Fetch the dataset using Scikit-learn
df = pd.DataFrame(california.data, columns=california.feature_names)   # Create a Pandas DataFrame with feature data
df['Target (MedHouseValue)'] = california.target   # Add the target variable (median house value) to the DataFrame

# Create a distribution plot of the target variable 'Target (MedHouseValue)'
sns.histplot(df['Target (MedHouseValue)'])   # Use a histogram to visualize the distribution of the target variable 'Target (MedHouseValue)'
plt.show()   # Display the distribution plot



# 18. Pairwise scatter plot of selected features:
# Interpretation: The pairplot (subset) shows scatter plots of selected numerical features against each other. 
# It allows you to focus on specific feature relationships
# Import necessary libraries
import seaborn as sns   # Import Seaborn for data visualization
import matplotlib.pyplot as plt   # Import Matplotlib for plot rendering
from sklearn.datasets import fetch_california_housing   # Import the California housing dataset from Scikit-learn
import pandas as pd   # Import Pandas for data manipulation and analysis

# Load the California housing dataset
california = fetch_california_housing()   # Fetch the dataset using Scikit-learn
df = pd.DataFrame(california.data, columns=california.feature_names)   # Create a Pandas DataFrame with feature data
df['Target (MedHouseValue)'] = california.target   # Add the target variable (median house value) to the DataFrame

# Create a pairwise scatter plot of selected numerical features
sns.pairplot(df[['MedInc', 'HouseAge', 'AveRooms']])   # Use pairplot to visualize pairwise scatter plots between 'MedInc', 'HouseAge', and 'AveRooms'
plt.show()   # Display the generated pairplot



# 19. Rug plot along the x-axis:
# Interpretation: A rug plot displays small vertical lines (ticks) along the x-axis, indicating the data points' positions. 
# It provides an overview of data distribution and density. The density of ticks at different positions shows the data concentration.
# Import necessary libraries
import seaborn as sns   # Import Seaborn for data visualization
import matplotlib.pyplot as plt   # Import Matplotlib for plot rendering
from sklearn.datasets import fetch_california_housing   # Import the California housing dataset from Scikit-learn
import pandas as pd   # Import Pandas for data manipulation and analysis

# Load the California housing dataset
california = fetch_california_housing()   # Fetch the dataset using Scikit-learn
df = pd.DataFrame(california.data, columns=california.feature_names)   # Create a Pandas DataFrame with feature data
df['Target (MedHouseValue)'] = california.target   # Add the target variable (median house value) to the DataFrame

# Create a rug plot along the x-axis for the 'MedInc' (Median Income) feature
sns.rugplot(df['MedInc'])   # Use a rug plot to visualize the distribution of the 'MedInc' (Median Income) feature along the x-axis
plt.show()   # Display the rug plot


# 20. Heatmap

# Interpreting a heatmap involves understanding the patterns and relationships between variables in a dataset based on their correlation values. 
# Heatmaps are useful graphical representations of correlation matrices and can provide valuable insights into the data. Here's how to interpret a heatmap:

# Color Intensity: In a heatmap, color intensity represents the strength of the correlation between two variables.
# Generally, warmer colors (e.g., red) indicate a positive correlation, while cooler colors (e.g., blue) represent a negative correlation. The more intense the color, the stronger the correlation.
# Diagonal Line: The diagonal line in the heatmap represents the correlation of each variable with itself, which is always 1. This is because a variable is perfectly correlated with itself.
# Symmetry: Heatmaps are symmetric around the diagonal. The correlation between variable A and variable B is the same as the correlation between variable B and variable A. This is due to the mathematical nature of correlation coefficients.
# Clustered Patterns: Similar variables tend to cluster together in a heatmap. High correlations between variables are represented by clustered patterns of the same or similar colors.
# Positive Correlation: If two variables have a positive correlation (closer to 1), they tend to increase together. In a heatmap, you'll observe a cluster of warm-colored cells.
# Negative Correlation: If two variables have a negative correlation (closer to -1), they tend to have an inverse relationship, i.e., one increases as the other decreases. In a heatmap, you'll observe a cluster of cool-colored cells.
# No Correlation: If two variables have no correlation (close to 0), the heatmap will show a lack of color intensity, indicating no clear relationship between the variables.
# Correlation Strength: The color bar on the side of the heatmap provides a legend for the correlation values. It helps you gauge the strength of the correlation based on color intensity.
# Annotations: Heatmaps often have numeric annotations within each cell, representing the actual correlation coefficient value. These values give you precise information about the strength and direction of the correlation
# Import necessary libraries
import seaborn as sns   # Import Seaborn for data visualization
import matplotlib.pyplot as plt   # Import Matplotlib for plot rendering
from sklearn.datasets import fetch_california_housing   # Import the California housing dataset from Scikit-learn
import pandas as pd   # Import Pandas for data manipulation and analysis

# Load the California housing dataset
california = fetch_california_housing()   # Fetch the dataset using Scikit-learn
df = pd.DataFrame(california.data, columns=california.feature_names)   # Create a Pandas DataFrame with feature data
df['Target (MedHouseValue)'] = california.target   # Add the target variable (median house value) to the DataFrame

# Calculate the correlation matrix
correlation_matrix = df.corr()   # Compute the correlation between numerical features in the DataFrame using the `.corr()` method

# Create a heatmap for the correlation matrix
plt.subplots(figsize=(18, 10))   # Set the size of the heatmap using `plt.subplots` and the `figsize` parameter
sns.heatmap(correlation_matrix, annot=True, annot_kws={'size': 14})   # Generate the heatmap with annotations to display the correlation values, and set the size of the annotations
plt.show()   # Display the heatmap


# Train Test Split Modelling

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = fetch_california_housing()
# Assuming 'data' is the DataFrame containing the California housing dataset, but there is no 'Price' column
# Train Test Split Modelling
X = data.drop(['Price'], axis=1)   # Extract the features by dropping the 'Price' column
X.head()   # Display the first few rows of the features DataFrame

y = data['Price']   # Extract the target variable 'Price' from the DataFrame
y.head()   # Display the first few rows of the target variable

# Split the data into training and testing sets with a test size of 20% and a random seed of 0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Display the shape of the training and testing data to check the number of samples and features
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# Create a Linear Regression model instance
model = LinearRegression()

# Fit the Linear Regression model to the training data
model.fit(X_train, y_train)

# Use the trained model to predict the target values for the test data
y_predict = model.predict(X_test)

# Display the predicted target values and the actual target values (y_test)
y_predict, y_test


# How to Evaluate the Regression Model Performance

# R-squared (R2) Score:
# R-squared is a statistical measure that indicates how well the regression model fits the data.
# R2 ranges from 0 to 1, where 0 indicates that the model does not explain any variance in the target variable (poor fit), and 1 indicates that the model perfectly explains the variance in the target variable (perfect fit).
# A higher R2 value suggests that a larger proportion of the variance in the target variable is explained by the model, indicating a better fit.
# However, R2 alone does not provide information about the quality of predictions or whether the model is overfitting or underfitting.

# Mean Absolute Error (MAE):
# MAE represents the average absolute difference between the true target values and the predicted target values.
# It measures the magnitude of errors made by the model on average.
# A lower MAE value indicates that the model's predictions are, on average, closer to the actual target values, suggesting better performance.

# Mean Squared Error (MSE):
# MSE is the average squared difference between the true target values and the predicted target values.
# It penalizes larger errors more than smaller errors due to the squaring operation.
# Like MAE, a lower MSE value indicates better performance, with smaller errors in the model's predictions.

#Root Mean Squared Error (RMSE):
# RMSE is the square root of MSE and is used to provide a more interpretable measure of the model's error.
# It is in the same unit as the target variable, making it easier to understand in the context of the problem domain.
# Similar to MSE, a lower RMSE value indicates better performance and suggests that the model's predictions are, on average, closer to the true target values

# After evaluating the regression model using these metrics, the code proceeds to visualize the true target values (y_test) and the predicted target values (y_predict) in a line plot using matplotlib. 
# This plot allows us to visually compare how well the model's predictions align with the actual target values. 
# The red line represents the true target values, and the green line represents the predicted target values. 
# If the green line follows the red line closely, it indicates that the model's predictions are accurate. 
# On the other hand, if there is significant deviation between the two lines, the model may not be performing well on the test set

# Import necessary libraries
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Assuming 'y_test' and 'y_predict' are the true target values and predicted target values, respectively

# Calculate the R-squared (coefficient of determination) to evaluate the model performance
r2_score(y_test, y_predict)

# Calculate the mean absolute error (MAE) to evaluate the model performance
mean_absolute_error(y_test, y_predict)

# Calculate the mean squared error (MSE) to evaluate the model performance
mean_squared_error(y_test, y_predict)

# Calculate the root mean squared error (RMSE) to evaluate the model performance
np.sqrt(mean_squared_error(y_test, y_predict))

# Assuming 'df' is the DataFrame containing the California housing dataset

# Display the DataFrame 'df'
df

# Create a plot with a figure size of (16, 6) for visualizing the predicted and true target values
plt.subplots(figsize=(16, 6))

# Create a list of x points from 0 to the length of 'y_test'
x_points = list(range(len(y_test)))

# Plot the true target values ('y_test') on the graph with a label 'y_true'
plt.plot(x_points, y_test, label='y_true')

# Plot the predicted target values ('y_predict') on the graph with a label 'y_predict'
plt.plot(x_points, y_predict, label='y_predict')

# Add a legend to the plot to distinguish between the true and predicted values
plt.legend()

# Display the plot
plt.show()


# Plotting Learning Curves
# In machine learning, the learning curve is a plot that shows how the performance of a model improves as the amount of training data increases. 
# It helps us understand how well the model is generalizing to unseen data and whether it is suffering from overfitting or underfitting. 
# The "Plotting Learning Curves" code for linear regression in Python aims to visualize the learning curve for a linear regression model using the California housing dataset.

# Interpreting the Learning Curve:

# Training Score (Red Line): The red line in the learning curve represents the performance of the model on the training data as the size of the training set increases. 
# If the training score is high and close to 1, it indicates that the model is fitting the training data well. 
# However, if the training score is low and starts to plateau, it suggests that the model is not learning well from the training data and may be underfitting.

# Test Score (Green Line): The green line in the learning curve represents the performance of the model on the test data as the size of the training set increases.
# The test score measures how well the model generalizes to unseen data. 
# If the test score is high and close to the training score, it indicates that the model is generalizing well to new data. 
# However, if the test score is significantly lower than the training score, it suggests that the model is overfitting the training data and not generalizing well.

# Convergence: In the learning curve, if both the training and test scores converge and stabilize as the size of the training set increases, it suggests that the model is likely to perform well on new data. 
# The convergence indicates that the model is learning effectively from the data and generalizing well.

# Gap between Training and Test Scores: If there is a substantial gap between the training and test scores, it indicates a variance problem. 
# A large gap suggests overfitting, while a small gap suggests that the model may be underfitting.

# Learning Curve Shape: The shape of the learning curve can give additional insights. 
# For example, if both training and test scores are low and close to each other, it suggests that the model is not learning well from the data, and collecting more data may not significantly improve performance. 
# On the other hand, if the training score is high, and the test score increases with the size of the training set, it indicates that the model would likely benefit from more data.

# By analyzing and interpreting the learning curve, we can make informed decisions on how to improve the model's performance, such as adjusting the model complexity, collecting more data, or applying regularization techniques to prevent overfitting

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, ShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing

# Load the California housing dataset
california = fetch_california_housing()
X = california.data
y = california.target

# Define a function to plot the learning curves
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, train_size=np.linspace(0.1, 1, 10)):
    # Create a new figure for the learning curve plot
    plt.figure()
    plt.title(title)   # Set the title of the plot
    plt.xlabel('Training Examples')   # Set the label for the x-axis
    plt.ylabel('Score')   # Set the label for the y-axis

    # Generate learning curves using the learning_curve function
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, train_sizes=train_size)

    # Calculate the mean and standard deviation of the training scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)

    # Calculate the mean and standard deviation of the test scores
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot the learning curve with filled areas indicating the standard deviation
    plt.grid()   # Add a grid to the plot
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color='red')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color='green')

    # Plot the training and test scores
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training Score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Test Score')

    # Add a legend to the plot to distinguish between training and test scores
    plt.legend(loc='best')

    # Return the plot
    return plt

# Define the title for the learning curve plot
title = 'Learning Curve for Linear Regression'

# Create a cross-validation strategy (ShuffleSplit in this case) for generating learning curves
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

# Create a LinearRegression model
model = LinearRegression()

# Call the 'plot_learning_curve' function to plot the learning curve
plot_learning_curve(model, title, X, y, ylim=(0.7, 1.01), cv=cv)

# Display the plot
plt.show()


# Machine Learning Model Interpretability
# Machine Learning Model Interpretability refers to the ability to understand and explain how a machine learning model makes predictions or decisions based on input data. 
# Interpretability is crucial for building trust and confidence in machine learning models, especially in high-stakes applications like healthcare, finance, and legal systems, where the model's decisions can have significant real-world consequences

# 1. Residual Plots
# Import necessary libraries
# !pip install -U yellowbrick   # Install or upgrade Yellowbrick library
from yellowbrick.regressor import ResidualsPlot, PredictionError   # Import ResidualsPlot and PredictionError visualizers from Yellowbrick
import matplotlib.pyplot as plt   # Import Matplotlib for plot rendering

# Create a ResidualsPlot visualizer
viz = ResidualsPlot(model)   # Initialize the ResidualsPlot visualizer with the trained model

# Fit the ResidualsPlot visualizer with training data and visualize the residuals
viz.fit(X_train, y_train)   # Fit the visualizer to the training data
viz.score(X_test, y_test)   # Score the visualizer on the test data to compute R-squared
viz.show()   # Display the ResidualsPlot

# Display the first plot
plt.show()
# Prediction Error Plot
# Create a PredictionError visualizer
viz = PredictionError(model)   # Initialize the PredictionError visualizer with the trained model

# Fit the PredictionError visualizer with training data and visualize the predictions
viz.fit(X_train, y_train)   # Fit the visualizer to the training data
viz.score(X_test, y_test)   # Score the visualizer on the test data to compute R-squared
viz.show()   # Display the PredictionError

# Display the second plot
plt.show()
