# 📦 Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# ---------------------------------------
# 📥 Step 1: Load and Clean Dataset
# ---------------------------------------
df = pd.read_csv("athlete_events-1.csv")

# Replace missing 'Medal' values with 'No Medal'
df['Medal'] = df['Medal'].fillna('No Medal')

# Replace missing Age with median
df['Age'] = df['Age'].fillna(df['Age'].median())

# Replace missing Height and Weight with sport-wise median
df['Height'] = df.groupby('Sport')['Height'].transform(lambda x: x.fillna(x.median()))
df['Weight'] = df.groupby('Sport')['Weight'].transform(lambda x: x.fillna(x.median()))
print(df.info())
print(df.describe())

# Encode Medal as numeric value
medal_map = {'No Medal': 0, 'Bronze': 1, 'Silver': 2, 'Gold': 3}
df['MedalValue'] = df['Medal'].map(medal_map)

# Create subset of medal winners
winners_df = df[df['Medal'] != 'No Medal']

# Set visual style
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)

# ---------------------------------------
# 📊 Step 2: EDA Visualizations
# ---------------------------------------

# 1. Medal Distribution (All Athletes)
plt.figure()
sns.countplot(data=df, x="Medal", order=["Gold", "Silver", "Bronze", "No Medal"], palette="Set2")
plt.title("Overall Medal Outcome")
plt.xlabel("Medal")
plt.ylabel("Number of Athletes")
plt.show()

# 2. Gender Distribution
plt.figure()
sns.countplot(data=df, x="Sex", palette="pastel")
plt.title("Gender Distribution of Athletes")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.xticks([0, 1], ["Male", "Female"])
plt.show()

# 3. Age Distribution
plt.figure()
sns.histplot(df['Age'], bins=30, kde=True, color="gray")
plt.title("Age Distribution of Athletes")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# 4. Medals Over the Years
plt.figure()
sns.countplot(data=winners_df, x="Year", color="steelblue")
plt.title("Total Medals Won Over Years")
plt.xlabel("Year")
plt.ylabel("Number of Medals")
plt.xticks(rotation=45)
plt.show()

# 5. Top 10 Sports by Medal Count
top_sports = winners_df['Sport'].value_counts().head(10)
plt.figure()
sns.barplot(x=top_sports.values, y=top_sports.index, palette="coolwarm")
plt.title("Top 10 Sports by Medal Count")
plt.xlabel("Number of Medals")
plt.ylabel("Sport")
plt.show()

# 6. Top 10 Countries by Medal Count
top_countries = winners_df['NOC'].value_counts().head(10)
plt.figure()
sns.barplot(x=top_countries.values, y=top_countries.index, palette="viridis")
plt.title("Top 10 Countries by Medal Count")
plt.xlabel("Number of Medals")
plt.ylabel("Country")
plt.show()

# ---------------------------------------
# 🔥 Step 3: Correlation Heatmap
# ---------------------------------------

# Select numerical columns
corr_df = df[['Age', 'Height', 'Weight', 'MedalValue']]
corr_matrix = corr_df.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, square=True)
plt.title('Correlation Heatmap')
plt.show()

# ---------------------------------------
# 📈 Step 4: Simple Linear Regression (Age ➝ MedalValue)
# ---------------------------------------

# Features and target
X = df[['Age']]
y = df['MedalValue']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate MSE
mse = mean_squared_error(y_test, y_pred)

# Output results
print(f"\n📉 Mean Squared Error (MSE): {mse:.4f}")
print(f"🧮 Coefficient: {model.coef_[0]:.4f}")
print(f"📐 Intercept: {model.intercept_:.4f}")

# Plot regression
plt.figure(figsize=(8, 5))
sns.scatterplot(x=X_test['Age'], y=y_test, alpha=0.3, label='Actual', color='gray')
sns.lineplot(x=X_test['Age'], y=y_pred, color='blue', label='Predicted')
plt.title('Linear Regression: Age vs MedalValue')
plt.xlabel('Age')
plt.ylabel('MedalValue')
plt.legend()
plt.show()
