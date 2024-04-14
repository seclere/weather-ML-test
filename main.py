import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('202403_CombinedData.csv')

sampled_data = data.sample(frac=0.1, random_state=42)  # Adjust the fraction as needed

# Define hyperparameters grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

# Initialize Gradient Boosting model
model = GradientBoostingRegressor(random_state=42)

X = sampled_data[['visibility', 'coord.lon', 'coord.lat', 'main.temp', 'main.pressure', 'main.humidity', 'wind.speed', 'clouds.all']]
y = sampled_data['main.temp_max']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Get best hyperparameters and retrain the model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)
print('Best Hyperparameters:', best_params)



# Apply filtering and smoothing to temperature data
filtered_smoothed_temp = sampled_data['main.temp'].clip(lower=sampled_data['main.temp'].quantile(0.05), upper=sampled_data['main.temp'].quantile(0.95))
filtered_smoothed_temp = filtered_smoothed_temp.rolling(window=7).mean()  # Adjust the window size for smoothing

# Plot the filtered and smoothed temperature trends
plt.figure(figsize=(10, 6))
plt.plot(sampled_data['datetime'], filtered_smoothed_temp, label='Filtered & Smoothed Temperature', color='green')
plt.xlabel('Datetime')
plt.ylabel('Temperature')
plt.title('Filtered & Smoothed Temperature Trends')
plt.legend()
plt.show()

# Visualize Humidity Variation
plt.figure(figsize=(10, 6))
sns.histplot(sampled_data['main.humidity'], bins=20, kde=True)
plt.xlabel('Humidity')
plt.ylabel('Frequency')
plt.title('Humidity Variation')
plt.show()

# Visualize Wind Speed Distribution
plt.figure(figsize=(10, 6))
sns.boxplot(sampled_data['wind.speed'])
plt.xlabel('Wind Speed')
plt.title('Wind Speed Distribution')
plt.show()

# Visualize Cloud Cover
plt.figure(figsize=(10, 6))
plt.scatter(sampled_data['clouds.all'], sampled_data['visibility'])
plt.xlabel('Cloud Cover')
plt.ylabel('Visibility')
plt.title('Cloud Cover vs Visibility')
plt.show()

# Visualize Rainfall Analysis (if rain data available)
if 'rain.1h' in sampled_data.columns:
    plt.figure(figsize=(10, 6))
    plt.hist(sampled_data['rain.1h'], bins=20)
    plt.xlabel('Rainfall (mm)')
    plt.ylabel('Frequency')
    plt.title('Rainfall Analysis')
    plt.show()
else:
    print('Rainfall data not available in the dataset.')
