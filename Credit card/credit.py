#libraries

import pandas as pd #handling datasets
import numpy as np #numerical operations

import matplotlib.pyplot as plt #data visualization & for graphs
import seaborn as sns #machine learning 

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import RandomOverSampler

from sklearn.metrics import(
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

# to load & check the dataset info

df = pd.read_csv("creditcard.csv")
print(df.head())

print(df.shape)

print(df.info())

print(df.isnull().sum())

#for understanding the target values

df["Class"]

print(df["Class"].value_counts())

sns.countplot(x="Class", data=df)

plt.title("Fraud vs Genuine Transactions")

plt.show()

#this features the scaling

scaler = StandardScaler()

df["Amount"] = scaler.fit_transform(df[["Amount"]])

#seperate the features & target

x = df.drop("Class", axis=1)

y = df["Class"]

#split dataset

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
) 

lr_model = LogisticRegression(max_iter=1000) #simpel logistic regression

lr_model.fit(x_train, y_train)

y_pred_lr = lr_model.predict(x_test) #to make predictions on the test set

#Evaluating the logistic regression 

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr))

precision = precision_score(y_test, y_pred_lr)
recall = recall_score(y_test, y_pred_lr)
f1 = f1_score(y_test, y_pred_lr)

print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(x_train, y_train)

y_pred_rf = rf_model.predict(x_test)

print("Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

ros = RandomOverSampler(random_state=42)
x_resampled, y_resampled = ros.fit_resample(x_train, y_train)
print(y_resampled.value_counts())

rf_balanced =RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf_balanced.fit(x_resampled, y_resampled)
y_pred_rf_balanced = rf_balanced.predict(x_test)
print("Balanced Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf_balanced))
print("\nBalanced Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf_balanced))

importance = rf_balanced.feature_importances_
feature_importance = pd.DataFrame({
    "Feature": x.columns,
    "Importance": importance
})
feature_importance = feature_importance.sort_values(
by="Importance", ascending=False
)

print(feature_importance.head(10))

plt.figure(figsize=(10, 6))

sns.barplot(
    x="Importance",
    y="Feature",
    data=feature_importance.head(10)
)
plt.title("Top 10 Important Features")
plt.show()