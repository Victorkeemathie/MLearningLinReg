# Data Preprocessing for Linear Regression

# Linear Model Assumptions

# 1. Linear Relationship: The assumption of a linear relationship between the independent variables (X) and the target variable (Y) means that the change in Y is directly proportional to the change in X. 
# In other words, the effect of each independent variable on the target variable is linear. 
# This assumption allows the model to estimate the coefficients for each independent variable to quantify their impact on the target variable.

# 2. Normally Distributed Independent Variables: This assumption states that the independent variables X should follow a normal distribution. 
# Normality is important for hypothesis testing and statistical inference.
# Deviation from normality may affect the validity of statistical tests and confidence intervals.

# 3. No Collinearity: Collinearity refers to the situation when two or more independent variables are highly correlated with each other. 
# High collinearity can lead to unstable estimates of the coefficients and makes it difficult to determine the individual effect of each independent variable on the target variable.

# 4. Homoscedasticity (Homogeneity of Variance): Homoscedasticity means that the variance of the residuals (the difference between the predicted and actual values of the target variable) is constant across all levels of the independent variables. 
# In other words, the spread of the residuals should be consistent along the entire range of X. Homoscedasticity ensures that the model's errors are equally distributed and do not change systematically as X changes.

# Importing important Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

import scipy.stats as stats
# scipy.stats: Scipy is a Python library that provides functions for scientific and technical computing. 
# The stats module within Scipy contains a wide range of statistical functions and tools. 
# In data preprocessing for linear regression, this library can be used for hypothesis testing, computing statistical metrics, and checking assumptions like normality of data, which is crucial for the correctness of linear regression models
from sklearn.datasets import fetch_california_housing
# fetch_california_housing: This is a function from Scikit-learn that allows you to load the California housing dataset. 
# The California housing dataset is a commonly used dataset in machine learning and contains various features related to housing prices in California.
# In data preprocessing, loading an appropriate dataset is essential to perform training and testing of the linear regression model.
from sklearn.linear_model import LinearRegression
# LinearRegression: This is a class from Scikit-learn that represents a linear regression model. 
# Linear regression is a fundamental statistical method used for modeling the relationship between a dependent variable and one or more independent variables. 
# In data preprocessing, the creation and training of the linear regression model are essential steps to fit the model to the data
from sklearn.model_selection import train_test_split
# train_test_split: This function from Scikit-learn is used to split the dataset into training and testing sets. 
# In data preprocessing, this step is critical to assess the model's generalization performance on unseen data. 
# By separating data into training and testing sets, we can evaluate how well the linear regression model can predict outcomes on new, unseen data
from sklearn.preprocessing import StandardScaler
# sklearn.preprocessing.StandardScaler: The StandardScaler is a preprocessing technique that scales the data by removing the mean and scaling to unit variance.
# This step is essential for many machine learning algorithms, including linear regression, as it helps in mitigating the impact of features with different scales. 
# By scaling the data, all features contribute equally to the model's learning process, preventing any particular feature from dominating the others
from sklearn.metrics import mean_squared_error
# from sklearn.metrics import mean_squared_error: This import statement brings in the mean_squared_error function from scikit-learn's metrics module. 
# The mean_squared_error function is used to calculate the mean squared error (MSE) between the actual target values and the predicted values obtained from a regression model.

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


# Data Preprocessing

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the California housing dataset
data = fetch_california_housing()
X = data.data  # Features
y = data.target  # Target variable (house prices)

# Simulate a relationship between a feature 'x' and the target 'y'
# For the sake of demonstration, let's assume a simple linear relationship:
# y = beta0 + beta1 * x + error
# where beta0 = 0 (intercept) and beta1 = 10 (coefficient), and error is random noise.
np.random.seed(0)  # Set random seed for reproducibility
n = len(X)  # Number of data points in the California housing dataset
x = np.random.randn(n)  # Generate random values for the feature 'x'
y_simulated = 10 * x + np.random.randn(n) * 2  # Simulate target variable 'y' with added noise

# Create a DataFrame with the simulated feature 'x' and the simulated target 'y'
demo_df = pd.DataFrame({'x': x, 'y': y_simulated})

# Fit a Linear Regression model to the simulated data
lr_model = LinearRegression()  # Create a Linear Regression model object
lr_model.fit(demo_df[['x']], demo_df['y'])  # Fit the model to the data

# Predict the target 'y' values using the trained Linear Regression model
y_predicted = lr_model.predict(demo_df[['x']])

# Calculate the Mean Squared Error (MSE) to evaluate the model's performance
mse = mean_squared_error(demo_df['y'], y_predicted)

# Print the simulated data and model performance
print("Simulated data and model performance:")
print("Simulated Data (First 5 rows):")
print(demo_df.head())
print("\nModel Coefficients:")
print("Intercept (beta0):", lr_model.intercept_)
print("Coefficient (beta1):", lr_model.coef_[0])
print("\nMean Squared Error (MSE):", mse)



# 1. Check Linear Assumption for California Dataset 

# Linearity:
# In scatter plots, linearity can be observed when the data points follow a roughly straight line pattern. 
# A linear relationship indicates that the two variables have a constant rate of change and a consistent direction.
# To assess linearity, visually inspect the scatter plot. If the points form a linear pattern, a linear regression model might be suitable for modeling the relationship between the variables.
# When using seaborn's lmplot, specifying order=1 fits a linear regression line to the data. If the data points align well with this line, it supports the linearity assumption. 
# However, if the data points do not align with the linear regression line, it suggests non-linearity.

#Non-Linearity:
# Non-linearity is evident when the scatter plot does not form a straight line, and the data points exhibit complex curves or patterns.
# If the scatter plot shows a clear curvature or the data points deviate significantly from the linear regression line (e.g., order=1), it suggests that a linear model may not be appropriate for capturing the relationship between the variables.
# In such cases, consider using higher-degree polynomial regression or other non-linear regression models to better capture the underlying pattern
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression

# Load the California housing dataset
data = fetch_california_housing()
X = data.data  # Features
y = data.target  # Target variable (house prices)

# Create a DataFrame with the features and target from the California housing dataset
california_df = pd.DataFrame(X, columns=data.feature_names)
california_df['Price'] = y  # Adding the target variable 'Price' to the DataFrame

# Check Linear Assumption for the California Dataset using different features
# Feature 'MedInc' vs. Target 'Price' with order=1 (Linear)
sns.lmplot(x='MedInc', y='Price', data=california_df, order=1)
plt.ylabel('Target (Price)')
plt.xlabel('Feature (MedInc)')
plt.title('Linear Relationship: MedInc vs. Price')
plt.show()

# Feature 'MedInc' vs. Target 'Price' with order=5 (Non-linear)
sns.lmplot(x='MedInc', y='Price', data=california_df, order=5)
plt.ylabel('Target (Price)')
plt.xlabel('Feature (MedInc)')
plt.title('Non-linear Relationship: MedInc vs. Price (order=5)')
plt.show()

# Feature 'AveRooms' vs. Target 'Price' with order=1 (Linear)
sns.lmplot(x='AveRooms', y='Price', data=california_df, order=1)
plt.ylabel('Target (Price)')
plt.xlabel('Feature (AveRooms)')
plt.title('Linear Relationship: AveRooms vs. Price')
plt.show()

# Feature 'AveOccup' vs. Target 'Price' with order=1 (Linear)
sns.lmplot(x='AveOccup', y='Price', data=california_df, order=1)
plt.ylabel('Target (Price)')
plt.xlabel('Feature (AveOccup)')
plt.title('Linear Relationship: AveOccup vs. Price')
plt.show()


# Variable Transformations

# Variable transformations are techniques used to alter the scale or shape of the data to make it more suitable for specific analysis or modeling purposes. 
# Each transformation has its own characteristics and use cases. Here's a brief explanatory overview of the mentioned variable transformations:

# 1. Logarithmic Transformation (Log Transformation) - np.log(X): / np.log1p
# The logarithmic transformation is used to reduce the magnitude of large values and compress the scale of data.
# It is commonly applied when the data has a right-skewed distribution, as it pulls in the extreme values closer to the center, making the distribution more symmetric.
# This transformation is particularly useful when the relationship between two variables is multiplicative rather than additive, as taking the logarithm converts the multiplicative relationship into an additive one.
# It is often used to stabilize variance in data, which is a common assumption in various statistical analyses.

# 2. Reciprocal Transformation (Reciprocal) - 1 / X:
# The reciprocal transformation is used to inverse the scale of data.
# It is commonly applied when the data has a left-skewed distribution with large values.
# Similar to the logarithmic transformation, the reciprocal transformation can help stabilize variance and make the distribution more symmetric.
# However, this transformation is sensitive to values close to zero, and if any data points are zero, you need to handle them carefully.

# 3. Square Root Transformation (Square Root) - sqrt(X):
# The square root transformation is used to moderate the effect of large values, similar to the logarithmic transformation.
# It is often applied when the data has a right-skewed distribution, pulling in extreme values closer to the center and making the distribution more symmetric.
# This transformation is less aggressive than the logarithmic transformation and can be a good alternative when the data has a wider range of values.

# 4. Exponential Transformation:
# The exponential transformation is used to reverse the effect of the logarithmic transformation.
# It is commonly applied when the data has been previously transformed using the logarithm and you need to revert to the original scale.
# It is useful in cases where the relationship between variables is multiplicative, as the exponential transformation converts the additive relationship back into a multiplicative one.

# 5. Box-Cox Transformation:
# The Box-Cox transformation is a family of power transformations that is used to stabilize variance and make the data more normally distributed.
# It is suitable for data that exhibits different degrees of skewness and has a positive range of values (non-negative).
# The transformation is parameterized by a lambda value (λ), which is estimated from the data to achieve the best transformation. This makes the Box-Cox transformation adaptive to the data's characteristics.

# 6. Yeo-Johnson Transformation:
# The Yeo-Johnson transformation is an extension of the Box-Cox transformation and can be applied to both positive and negative data values.
# Similar to the Box-Cox transformation, it stabilizes variance and normalizes the data distribution.
# The Yeo-Johnson transformation introduces an additional parameter to handle negative values, making it more flexible than the Box-Cox transformation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from scipy.stats import boxcox, yeojohnson

# Load the California housing dataset
data = fetch_california_housing()
X = data.data  # Features
y = data.target  # Target variable (house prices)

# Create a DataFrame with the features and target from the California housing dataset
california_df = pd.DataFrame(X, columns=data.feature_names)
california_df['Price'] = y  # Adding the target variable 'Price' to the DataFrame

# Step 1: Log Transformation
california_df['Log_Transformed_Price'] = np.log1p(california_df['Price'])

# Plot the histogram for the Log Transformation
plt.figure(figsize=(8, 6))
plt.hist(california_df['Log_Transformed_Price'], bins=30)
plt.title('Log Transformed Price')
plt.xlabel('Log Transformed Price')
plt.ylabel('Frequency')
plt.show()

# Step 2: Reciprocal Transformation
california_df['Reciprocal_Transformed_Price'] = 1 / california_df['Price']

# Plot the histogram for the Reciprocal Transformation
plt.figure(figsize=(8, 6))
plt.hist(california_df['Reciprocal_Transformed_Price'], bins=30)
plt.title('Reciprocal Transformed Price')
plt.xlabel('Reciprocal Transformed Price')
plt.ylabel('Frequency')
plt.show()

# Step 3: Square Root Transformation
california_df['Sqrt_Transformed_Price'] = np.sqrt(california_df['Price'])

# Plot the square root transformation using sns.lmplot
sns.lmplot(x='Sqrt_Transformed_Price', y='Price', data=california_df, order=1)
plt.title('Square Root Transformed Price')
plt.xlabel('Square Root Transformed Price')
plt.ylabel('Price')
plt.show()

# Step 4: Exponential Transformation
california_df['Exponential_Transformed_Price'] = np.exp(california_df['Price'])

# Plot the histogram for the Exponential Transformation
plt.figure(figsize=(8, 6))
plt.hist(california_df['Exponential_Transformed_Price'], bins=30)
plt.title('Exponential Transformed Price')
plt.xlabel('Exponential Transformed Price')
plt.ylabel('Frequency')
plt.show()

# Step 5: Box-Cox Transformation
california_df['BoxCox_Transformed_Price'], _ = boxcox(california_df['Price'] + 1)

# Plot the histogram for the Box-Cox Transformation
plt.figure(figsize=(8, 6))
plt.hist(california_df['BoxCox_Transformed_Price'], bins=30)
plt.title('Box-Cox Transformed Price')
plt.xlabel('Box-Cox Transformed Price')
plt.ylabel('Frequency')
plt.show()

# Step 6: Yeo-Johnson Transformation
california_df['YeoJohnson_Transformed_Price'], _ = yeojohnson(california_df['Price'])

# Plot the histogram for the Yeo-Johnson Transformation
plt.figure(figsize=(8, 6))
plt.hist(california_df['YeoJohnson_Transformed_Price'], bins=30)
plt.title('Yeo-Johnson Transformed Price')
plt.xlabel('Yeo-Johnson Transformed Price')
plt.ylabel('Frequency')
plt.show()

# Alternatively:
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.preprocessing import PowerTransformer

# Load the Boston housing dataset
boston = load_boston()
data = pd.DataFrame(boston.data, columns=boston.feature_names)
data['Price'] = boston.target

# Step 4: Box-Cox Transformation
transformer_boxcox = PowerTransformer(method='box-cox', standardize=False)
data['BoxCox_Transformed_Price'] = transformer_boxcox.fit_transform(data[['Price']])

# Step 5: Yeo-Johnson Transformation
transformer_yeojohnson = PowerTransformer(method='yeo-johnson', standardize=False)
data['YeoJohnson_Transformed_Price'] = transformer_yeojohnson.fit_transform(data[['Price']])

"""

# Checking Variables Normality
# Checking for normality in a variable is a crucial step in data analysis and modeling. 
# Normality refers to the distribution of data, where it follows a Gaussian or bell-shaped curve. 
# A normal distribution is essential for many statistical tests and modeling assumptions. 
# In Python, you can use various libraries and methods to assess the normality of a variable.
# Interpreting Normality:
# If the histogram and KDE curve show a symmetric, bell-shaped distribution, and the data points are concentrated around the mean, the variable is likely normally distributed.
# If the histogram is skewed, the variable might be non-normal. Positive skewness (right-skewed) indicates a long right tail, while negative skewness (left-skewed) indicates a long left tail.
# Extreme outliers in the data can also affect normality, causing the tails to stretch.
# Formal statistical tests like the Shapiro-Wilk test or Anderson-Darling test provide p-values to assess the significance of normality. 
# A p-value less than 0.05 indicates that the data significantly deviates from a normal distribution.
# Interpreting Q-Q plots:
# In a Q-Q plot for a normal distribution, the points should approximately follow a straight line.
# If the points deviate significantly from the straight line, it suggests that the data does not follow a normal distribution.
# Points curving upward or downward at the ends of the Q-Q plot indicate heavy tails and non-normality, respectively.
# A straight line in the middle but curving at the ends suggests light-tailedness

# 1. Using Displots: 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# Load the California housing dataset
data = fetch_california_housing()
X = data.data  # Features
y = data.target  # Target variable (house prices)

# Create a DataFrame with the features from the California housing dataset
california_df = pd.DataFrame(X, columns=data.feature_names)

# Step 1: Plot 'MedInc' feature
sns.displot(california_df['MedInc'], bins=30, kde=True)
plt.title('Distribution of MedInc')
plt.xlabel('MedInc')
plt.ylabel('Frequency')
plt.show()

# Step 2: Plot 'AveRooms' feature
sns.displot(california_df['AveRooms'], bins=30, kde=True)
plt.title('Distribution of AveRooms')
plt.xlabel('AveRooms')
plt.ylabel('Frequency')
plt.show()

# Step 3: Plot 'AveBedrms' feature
sns.displot(california_df['AveBedrms'], bins=30, kde=True)
plt.title('Distribution of AveBedrms')
plt.xlabel('AveBedrms')
plt.ylabel('Frequency')
plt.show()

# Step 4: Plot 'AveOccup' feature
sns.displot(california_df['AveOccup'], bins=30, kde=True)
plt.title('Distribution of AveOccup')
plt.xlabel('AveOccup')
plt.ylabel('Frequency')
plt.show()


# 2. Using Q-Q Plots

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.datasets import fetch_california_housing

# Load the California housing dataset
data = fetch_california_housing()
X = data.data  # Features
y = data.target  # Target variable (house prices)

# Create a DataFrame with the features from the California housing dataset
california_df = pd.DataFrame(X, columns=data.feature_names)

# Step 1: Plot Q-Q plot for 'MedInc' feature
plt.figure(figsize=(8, 6))
stats.probplot(california_df['MedInc'], dist='norm', plot=plt)
plt.title('Q-Q Plot for MedInc')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.show()

# Step 2: Plot Q-Q plot for 'RM' feature
plt.figure(figsize=(8, 6))
stats.probplot(california_df['RM'], dist='norm', plot=plt)
plt.title('Q-Q Plot for RM')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.show()

# Step 3: Plot Q-Q plot for 'LSTAT' feature
plt.figure(figsize=(8, 6))
stats.probplot(california_df['LSTAT'], dist='norm', plot=plt)
plt.title('Q-Q Plot for LSTAT')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.show()

# Step 4: Plot Q-Q plot for 'CRIM' feature
plt.figure(figsize=(8, 6))
stats.probplot(california_df['CRIM'], dist='norm', plot=plt)
plt.title('Q-Q Plot for CRIM')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.show()


# 3. 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Generate some example data
np.random.seed(0)
data = np.random.normal(loc=0, scale=1, size=1000)

# Step 1: Visual Inspection
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
sns.histplot(data, kde=True)
plt.title('Histogram')

plt.subplot(1, 2, 2)
stats.probplot(data, plot=plt)
plt.title('Q-Q Plot')

plt.tight_layout()
plt.show()

# Step 2: Statistical Tests
# Shapiro-Wilk Test
shapiro_test_statistic, shapiro_p_value = stats.shapiro(data)
print(f"Shapiro-Wilk Test - Statistic: {shapiro_test_statistic}, p-value: {shapiro_p_value}")

# Anderson-Darling Test
anderson_statistic, anderson_critical_values, anderson_significance_levels = stats.anderson(data)
print(f"Anderson-Darling Test - Statistic: {anderson_statistic}")
for i in range(len(anderson_critical_values)):
    sl = anderson_significance_levels[i]
    cv = anderson_critical_values[i]
    if anderson_statistic < cv:
        print(f"The data seems normal at {sl * 100:.1f}% significance level.")
    else:
        print(f"The data does not seem normal at {sl * 100:.1f}% significance level.")


# Variable Transformation for Normality
# Variable transformation for normality is a technique used to convert a non-normally distributed variable into one that approximates a normal distribution.
# A normal distribution (also known as Gaussian distribution) is characterized by a bell-shaped curve, and many statistical analyses and models assume normality. 
# Transforming variables to be more normally distributed can improve the validity and reliability of statistical tests and models. 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PowerTransformer

# Load the California housing dataset
data = fetch_california_housing()
X = data.data  # Features
y = data.target  # Target variable (house prices)

# Create a DataFrame with the features from the California housing dataset
california = pd.DataFrame(X, columns=data.feature_names)

# Step 1: Yeo-Johnson Transformation on 'MedInc' feature
transformer = PowerTransformer(method='yeo-johnson', standardize=False)
california['yj_MedInc'] = transformer.fit_transform(california[['MedInc']])

# Plot the histograms for 'MedInc' and 'yj_MedInc' features
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
sns.histplot(california['MedInc'], bins=30, kde=True)
plt.title('Distribution of MedInc')
plt.xlabel('MedInc')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.histplot(california['yj_MedInc'], bins=30, kde=True)
plt.title('Yeo-Johnson Transformed Distribution of MedInc')
plt.xlabel('Yeo-Johnson Transformed MedInc')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Q-Q plot for 'yj_MedInc' feature
plt.figure(figsize=(8, 6))
stats.probplot(california['yj_MedInc'], dist='norm', plot=plt)
plt.title('Q-Q Plot for Yeo-Johnson Transformed MedInc')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.show()


# Homoscedasticity
import numpy as np
import matplotlib.pyplot as plt

# Simulate data for a linear relationship (Homoscedasticity)
np.random.seed(0)
x_homoscedasticity = np.linspace(0, 10, 100)
y_homoscedasticity = 2 * x_homoscedasticity + np.random.normal(0, 2, 100)

# Fit a linear regression model
coeff_homoscedasticity = np.polyfit(x_homoscedasticity, y_homoscedasticity, 1)
y_pred_homoscedasticity = np.polyval(coeff_homoscedasticity, x_homoscedasticity)

# Calculate residuals
residuals_homoscedasticity = y_homoscedasticity - y_pred_homoscedasticity

# Plot the residual plot
plt.scatter(x=x_homoscedasticity, y=residuals_homoscedasticity)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot - Homoscedasticity')
plt.show()


# Simulate data for a non-linear relationship (Heteroscedasticity)
np.random.seed(0)
x_heteroscedasticity = np.linspace(0, 10, 100)
y_heteroscedasticity = 2 * x_heteroscedasticity + np.random.normal(0, x_heteroscedasticity, 100)

# Fit a linear regression model
coeff_heteroscedasticity = np.polyfit(x_heteroscedasticity, y_heteroscedasticity, 1)
y_pred_heteroscedasticity = np.polyval(coeff_heteroscedasticity, x_heteroscedasticity)

# Calculate residuals
residuals_heteroscedasticity = y_heteroscedasticity - y_pred_heteroscedasticity

# Plot the residual plot
plt.scatter(x=x_heteroscedasticity, y=residuals_heteroscedasticity)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot - Heteroscedasticity')
plt.show()


# eg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import mean_squared_error

# Load the California housing dataset
data = fetch_california_housing()
X = data.data  # Features
y = data.target  # Target variable (house prices)

# Create a DataFrame with the features from the California housing dataset
california = pd.DataFrame(X, columns=data.feature_names)

# Select 'MedInc' as the predictor variable (X) and 'Price' as the target variable (y)
X = california[['MedInc']]
y = data.target

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Step 1: Linear Regression on 'MedInc' feature
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate the residuals (error)
error = y_test - y_pred

# Plot the residuals against 'MedInc'
plt.scatter(x=X_test, y=error)
plt.xlabel('MedInc')
plt.ylabel('Residuals (Error)')
plt.title('Residual Plot for Linear Regression')
plt.show()

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print('MSE:', mse)

# Step 2: Yeo-Johnson Transformation on 'MedInc' feature
transformer_yj = PowerTransformer(method='yeo-johnson', standardize=False)
X_train_yj = transformer_yj.fit_transform(X_train)
X_test_yj = transformer_yj.transform(X_test)

model_yj = LinearRegression()
model_yj.fit(X_train_yj, y_train)
y_pred_yj = model_yj.predict(X_test_yj)

error_yj = y_test - y_pred_yj

plt.scatter(x=X_test_yj, y=error_yj)
plt.xlabel('Yeo-Johnson Transformed MedInc')
plt.ylabel('Residuals (Error)')
plt.title('Residual Plot for Yeo-Johnson Transformed MedInc')
plt.show()

mse_yj = mean_squared_error(y_test, y_pred_yj)
print('MSE (Yeo-Johnson):', mse_yj)

# Step 3: Logarithmic Transformation on 'MedInc' feature
X_train_log = np.log(X_train)
X_test_log = np.log(X_test)

model_log = LinearRegression()
model_log.fit(X_train_log, y_train)
y_pred_log = model_log.predict(X_test_log)

error_log = y_test - y_pred_log

plt.scatter(x=X_test_log, y=error_log)
plt.xlabel('Log(MedInc)')
plt.ylabel('Residuals (Error)')
plt.title('Residual Plot for Logarithmic Transformation')
plt.show()

mse_log = mean_squared_error(y_test, y_pred_log)
print('MSE (Logarithmic):', mse_log)

# Step 4: Reciprocal Transformation on 'MedInc' feature
X_train_reciprocal = 1 / X_train
X_test_reciprocal = 1 / X_test

model_reciprocal = LinearRegression()
model_reciprocal.fit(X_train_reciprocal, y_train)
y_pred_reciprocal = model_reciprocal.predict(X_test_reciprocal)

error_reciprocal = y_test - y_pred_reciprocal

plt.scatter(x=X_test_reciprocal, y=error_reciprocal)
plt.xlabel('1 / MedInc')
plt.ylabel('Residuals (Error)')
plt.title('Residual Plot for Reciprocal Transformation')
plt.show()

mse_reciprocal = mean_squared_error(y_test, y_pred_reciprocal)
print('MSE (Reciprocal):', mse_reciprocal)



# Multicollinearity
# Multicollinearity is a phenomenon that occurs when two or more predictor variables (also known as independent variables or features) in a regression model are highly correlated with each other. 
# In other words, multicollinearity arises when there is a strong linear relationship between two or more independent variables

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# Load the California housing dataset
data = fetch_california_housing()
california = pd.DataFrame(data.data, columns=data.feature_names)

# Display the first few rows of the dataset
california.head()

# Calculate the correlation matrix
corr_mat = california.corr()

# Plot a heatmap to visualize the correlation matrix
plt.subplots(figsize=(10, 10))
ax = sns.heatmap(data=corr_mat, annot=True)
plt.title('Correlation Matrix Heatmap')
plt.show()


# Normalization and Standardization

# Normalization (also known as Min-Max scaling) rescales the features to a specific range, typically between 0 and 1. 
# It maps the minimum value of the feature to 0 and the maximum value to 1, while preserving the relative distances between other data points
# Normalization is useful when the scale of features varies widely, and we want to bring them to a common scale. 
# It ensures that all features contribute equally to the model, preventing the dominance of one feature over others

# Standardization (also called z-score normalization) transforms the features to have zero mean and unit variance. 
# It subtracts the mean of the feature from each data point and then divides it by the standard deviation
# Standardization is suitable when the features have different units or different scales. It centers the data around 0 and scales it to have a variance of 1. 
# Standardized features are more interpretable, and the model can converge faster during training
