# lagistic-reggression
# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_csv(r"C:\Users\haree\OneDrive\Desktop\Titanic_train.csv")
df.head()
df.tail()
df.info()
df.describe()
columns = list(df[['PassengerId', 'Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']])

def describe_data(data, col):
    print ('\n\n', col)
    print ('_' * 40)
    print ('Mean:', np.mean(data)) #NumPy Mean
    print ('STD:', np.std(data))   #NumPy STD
    print ('Min:', np.min(data))   #NumPy Min
    print ('Max:', np.max(data))   #NumPy Max

for c in columns:
    describe_data(df[c], c)

    df.describe(include=['O'])
    # Select Sex and Survived Columns
# Group data by sex
# Show index for row
# Return mean of values for male/female
# Sort by highest survival rate first
df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sns.set(style='ticks', color_codes=True)

# Plot passenger age distribution
age_hist = sns.FacetGrid(df)
age_hist.map(plt.hist, 'Age', bins=20)
# Plot histogram of survival by age
age_hist = sns.FacetGrid(df, col='Survived', hue='Survived')
age_hist.map(plt.hist, 'Age', bins=15)# Plot histogram by survival, sex, and age
age_sex_hist = sns.FacetGrid(df, col='Survived', row='Sex', hue='Survived')
age_sex_hist.map(plt.hist, 'Age', bins=20)
#Create Passenger column to plot total passengers
df['Passenger'] = 'Passenger'
# Create Class column with string values for class
df['Class'] = df['Pclass'].map( {1: 'Upper', 2: 'Middle', 3: 'Lower'} )

# Create PointPlot for Passengers by Class
bp = sns.pointplot(x='Passenger', y='Survived', hue='Class', data=df, hue_order=['Lower', 'Middle', 'Upper'])
bp.set(ylabel='% Survivors', xlabel='Passenger Class')
plt.boxplot(df.PassengerId)
plt.show()
cols = df.columns
colours = ['blue', 'yellow'] # specify the colours - yellow is missing. blue is not missing.
sns.heatmap(df[cols].isnull(),cmap=sns.color_palette(colours))
sns.pairplot(df, kind="scatter", hue="Survived", palette="Set2",plot_kws=dict(s=80, edgecolor="black", linewidth=0.4))
plt.show()
df[df.isnull().any(axis=1)].head()
df.isnull().sum()
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
dff = df.copy()
for feat in df.columns:
    dff[feat] = dff[feat].astype('category')
    dff[feat] = dff[feat].cat.codes
dff.head()
print(df.head())

# Check for missing values and handle them if necessary
print(df.isnull().sum())

# Drop rows with missing values for simplicity (not recommended in practice)
df.dropna(inplace=True)

# Encode categorical variables (if any)
df_encoded = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Pclass'], drop_first=True)

# Select features and target variable
X = df_encoded.drop(['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
y = df_encoded['Survived']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features if needed
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build and train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix:\n{conf_matrix}')

class_report = classification_report(y_test, y_pred)
print(f'Classification Report:\n{class_report}')
df.dropna(subset=['Age', 'Embarked'], inplace=True)

# Encode categorical variables
df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Pclass'], drop_first=True)

# Select features and target variable
X = df.drop(['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
y = df['Survived']

# Step 3: Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Build and train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 5: Interpret coefficients and discuss feature significance

# Coefficients and intercept
coefficients = pd.DataFrame({'feature': X.columns, 'coefficient': model.coef_[0]})
intercept = model.intercept_[0]

print(f'Intercept: {intercept:.4f}')
print(coefficients)

# Feature significance discussion
# Interpret coefficients in terms of their impact on the log-odds of survival probability
# Positive coefficients increase the log-odds (increasing likelihood of survival)
# Negative coefficients decrease the log-odds (decreasing likelihood of survival)









