import pandas as pd
import requests

# Download data
url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/diabetes_scale'
response = requests.get(url)

# Split the data by row
lines = response.text.splitlines()

# Decomposing data
parsed_data = []
for line in lines:
    parts = line.split()  
    row = {"label": parts[0]}  
    
    
    for item in parts[1:]:
        key_value = item.split(':')  
        if len(key_value) == 2:  
            key, value = key_value
            row[key] = float(value)  
    parsed_data.append(row)

# transform to DataFrame
df = pd.DataFrame(parsed_data)

# Index rows starting at 1
df.index = df.index + 1

# Add variable name (column name)
df.columns = ['outcome', 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Diabetes pedigree', 'Age']

# Replace all the -1's in the outcome column with 0's
df['outcome'] = df['outcome'].replace({-1: 0})

# Check replacement result
print(df['outcome'].value_counts())

# Data visualization
statistics = df.describe()
print(statistics)

# Check the null value in the DataFrame
print("Check null：")
print(df.isnull().sum())  # The number of hollow values per column

# Replace the value in the outcome column
df['outcome'] = df['outcome'].replace({'+1': 1, '-1': 0}).astype(int)


# Calculate the mean again without null values
mean_without_nan = df.mean()

print("The average without null values：")
print(mean_without_nan)

# Fill in the empty values with the average of each column
df = df.fillna(mean_without_nan)

# View the filled data
print("The DataFrame after the null value is filled：")
print(df)

# Check the DataFrame for null values
missing_values = df.isnull().sum()

# Outputs the number of null values for each column
print("The number of empty values remaining in each column：")
print(missing_values)

# Check the type of data in each column
print("Check the type of data in each column：")
print(df.dtypes)

import matplotlib.pyplot as plt

# Create a 2x4 subgraph layout
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Variable list
variables = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
             'Insulin', 'BMI', 'Diabetes pedigree', 'Age']

# Traverse the variables and draw each subgraph
for i, var in enumerate(variables):
    row = i // 4  
    col = i % 4   
    axes[row, col].hist(df[var], bins=15, color='skyblue', edgecolor='black')  
    axes[row, col].set_title(f'Distribution of {var}')
    axes[row, col].set_xlabel(var)
    axes[row, col].set_ylabel('Frequency')
plt.tight_layout()

plt.show()

import seaborn as sns
# Suppose df is a DataFrame with 8 variables
# Calculate the correlation matrix
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))

# Draw a heat map of the correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of 8 Variables')
plt.show()

# Plot a pie chart for the 'outcome' column
plt.figure(figsize=(8, 6))

# Calculate the frequency distribution of the 'outcome'
outcome_counts = df['outcome'].value_counts()

plt.pie(outcome_counts, labels=outcome_counts.index, autopct='%1.1f%%', colors=['skyblue', 'lightgreen'], startangle=90)

plt.title('Outcome Distribution')
plt.show()

from sklearn.model_selection import train_test_split
import tensorflow as tf

# Suppose df is a loaded data box and contains the 'outcome' column as the target variable
# Separate feature and target variables
X = df.drop(columns=['outcome'])  
y = df['outcome']
# Divide the data into training set and test set, according to 80% training set, 20% test set allocation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Output the size of the training set and test set
print(f"Training set size: {X_train.shape}, Testing set size: {X_test.shape}")


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_curve, roc_auc_score,accuracy_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam
import numpy as np 


# Build an optimized multi-layer perceptron model
model = Sequential()
model.add(Dense(256, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.5))  # Prevent overfitting
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Prevent overfitting
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compilation model
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Training model
# Train the model using class weights and early stop mechanisms
history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.2)

# Evaluate the accuracy of the model
y_pred_proba = model.predict(X_test)[:,0]
# Probabilities are converted to binary labels, converted to 0 and 1
y_pred = (y_pred_proba > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)

# Output the accuracy on the test set
print(f"Test accuracy on the set: {accuracy:.4f}")

# Computation performance index
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"AUC:{auc:.4f}")



# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # 随机猜测的对角线
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Computational confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[1, -1], yticklabels=[1, -1])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (diabetics = 1, non-diabetics = -1)')
plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

# Calculate the Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)

# 计算 PR AUC
pr_auc = auc(recall, precision)

# Calculate the PR AUC
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()  

# Plot the learning rate curve
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()  

