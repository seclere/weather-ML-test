import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load data
data = pd.read_csv('202403_CombinedData.csv')

# Preprocess data (handle missing values, encoding, scaling, etc.)
# Assuming preprocessing steps have been done

# Split data into features and target variable
X = data[['visibility', 'coord.lon', 'coord.lat', 'main.temp', 'main.pressure', 'main.humidity', 'wind.speed', 'clouds.all']]
y = data['main.temp_max']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

# Visualize Temperature Trends
plt.figure(figsize=(10, 6))
plt.plot(data['datetime'], data['main.temp'], label='Temperature')
plt.xlabel('Datetime')
plt.ylabel('Temperature')
plt.title('Temperature Trends')
plt.legend()
plt.show()

# Visualize Humidity Variation
plt.figure(figsize=(10, 6))
sns.histplot(data['main.humidity'], bins=20, kde=True)
plt.xlabel('Humidity')
plt.ylabel('Frequency')
plt.title('Humidity Variation')
plt.show()

# Visualize Wind Speed Distribution
plt.figure(figsize=(10, 6))
sns.boxplot(data['wind.speed'])
plt.xlabel('Wind Speed')
plt.title('Wind Speed Distribution')
plt.show()

# Visualize Cloud Cover
plt.figure(figsize=(10, 6))
plt.scatter(data['clouds.all'], data['visibility'])
plt.xlabel('Cloud Cover')
plt.ylabel('Visibility')
plt.title('Cloud Cover vs Visibility')
plt.show()

# Visualize Rainfall Analysis (if rain data available)
if 'rain.1h' in data.columns:
    plt.figure(figsize=(10, 6))
    plt.hist(data['rain.1h'], bins=20)
    plt.xlabel('Rainfall (mm)')
    plt.ylabel('Frequency')
    plt.title('Rainfall Analysis')
    plt.show()
else:
    print('Rainfall data not available in the dataset.')
