# Iris Flower Classification

## 🌸 Project Overview

This project implements a **machine learning classification system** that accurately classifies iris flowers into three species (**Setosa**, **Versicolor**, and **Virginica**) based on their sepal and petal measurements.

The Iris dataset is one of the most popular datasets in machine learning and is perfect for learning classification algorithms.

## 📊 Dataset

The dataset contains 150 iris flower samples with the following measurements:
- **Sepal Length** (cm)
- **Sepal Width** (cm)
- **Petal Length** (cm)
- **Petal Width** (cm)
- **Species** (Target variable)

### Classes:
1. **Iris-setosa**
2. **Iris-versicolor**
3. **Iris-virginica**

## 🎯 Objective

Train and evaluate multiple machine learning models to:
1. Learn patterns from iris measurements
2. Accurately classify new iris flowers
3. Compare model performance
4. Visualize results and feature importance

## 🚀 Getting Started

### Prerequisites

Install required packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Running the Code

```bash
python iris.py
```

## 📈 Models Implemented

### 1. **Random Forest Classifier**
- Ensemble method combining multiple decision trees
- Handles non-linear relationships well
- Provides feature importance rankings
- Generally the most robust for this task

### 2. **Support Vector Machine (SVM)**
- Uses kernel methods to find optimal separation boundaries
- Effective for high-dimensional data
- Good generalization ability

### 3. **K-Nearest Neighbors (KNN)**
- Simple and intuitive algorithm
- Instance-based learning
- Effective for small-to-medium datasets

## 📋 Project Structure

```
IRIS-FLOWER/
├── IRIS.csv          # Dataset file
├── iris.py           # Main classification script
└── README.md         # This file
```

## 🔍 Key Features

### Data Preprocessing
- ✅ Data loading and exploration
- ✅ Class distribution analysis
- ✅ Feature standardization (StandardScaler)
- ✅ Stratified train-test split (80-20)

### Model Training
- ✅ Multiple classifier algorithms
- ✅ Hyperparameter optimization
- ✅ Parallel processing (Random Forest)

### Model Evaluation
- ✅ Accuracy, Precision, Recall, F1-Score
- ✅ Confusion matrices
- ✅ Classification reports
- ✅ Model comparison visualization

### Visualization
- ✅ Confusion matrices for all models
- ✅ Model performance comparison chart
- ✅ Feature importance plot
- ✅ Saved plots as PNG files

## 📊 Expected Output

When you run the script, you'll get:

1. **Data Summary**
   - Dataset shape and structure
   - Statistical information
   - Class distribution

2. **Model Results**
   - Training completion status
   - Accuracy, Precision, Recall, F1-Score for each model
   - Detailed classification reports

3. **Visualizations** (saved as PNG files)
   - `confusion_matrices.png` - Confusion matrices for all models
   - `model_comparison.png` - Performance metrics comparison
   - `feature_importance.png` - Feature importance ranking

4. **Best Model Identification**
   - The model with highest accuracy is identified
   - Example predictions on new iris flowers

## 🎓 Example Usage

### Making Predictions on New Data

```python
from iris import predict_new_iris, train_models, preprocess_data, split_data
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
df = pd.read_csv('IRIS.csv')
X_scaled, y, scaler, _ = preprocess_data(df)

# Train model
X_train, X_test, y_train, y_test = split_data(X_scaled, y)
models = train_models(X_train, y_train)
best_model = models['Random Forest']

# Predict new iris
new_measurements = [5.1, 3.5, 1.4, 0.2]  # [sepal_length, sepal_width, petal_length, petal_width]
prediction = predict_new_iris(best_model, scaler, new_measurements)
print(f"Predicted species: {prediction}")  # Output: setosa
```

## 📈 Performance Metrics Explained

- **Accuracy**: Percentage of correct predictions
- **Precision**: Of positive predictions, how many were correct
- **Recall**: Of actual positives, how many were identified
- **F1-Score**: Harmonic mean of precision and recall

## 🔧 Customization

You can modify the following in `iris.py`:

1. **Model Parameters**: Adjust `n_estimators`, `kernel`, `n_neighbors`, etc.
2. **Test Size**: Change the `test_size` parameter in `split_data()`
3. **Feature Scaling**: Use different scaling methods if needed
4. **Number of Neighbors**: Adjust `n_neighbors` in KNN
5. **Random State**: Change for different train-test splits

## 📚 Learning Outcomes

By studying this project, you'll learn:
- ✅ Data preprocessing and feature scaling
- ✅ Train-test split and stratification
- ✅ Training multiple classification models
- ✅ Model evaluation and comparison
- ✅ Data visualization with matplotlib and seaborn
- ✅ Using scikit-learn for machine learning
- ✅ Feature importance analysis
- ✅ Making predictions on new data

## 🎯 Real-World Applications

Similar classification techniques are used in:
- 🏥 Medical diagnosis
- 🐦 Species identification
- 📧 Spam detection
- 🎬 Movie recommendation
- 🏦 Credit risk assessment
- 🔐 Fraud detection

## 📝 Notes

- The dataset is already balanced across all three classes
- Feature scaling is important for SVM and KNN
- Random Forest doesn't require feature scaling but we apply it for consistency
- Results may vary slightly due to randomness - use `random_state=42` for reproducibility

## 🔗 References

- [Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Machine Learning Basics](https://www.deeplearningbook.org/)

## 📄 License

This project is open for educational purposes.

---

**Happy Learning! 🌸📊**
