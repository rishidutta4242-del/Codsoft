"""
Iris Flower Classification Model
==================================
This script trains machine learning models to classify iris flowers
into three species (setosa, versicolor, virginica) based on sepal
and petal measurements.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings('ignore')

# ==================== DATA LOADING ====================
def load_data(filepath):
    """Load iris dataset from CSV file."""
    df = pd.read_csv(filepath)
    print("Dataset shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())
    print("\nDataset info:")
    print(df.info())
    print("\nStatistical summary:")
    print(df.describe())
    return df

# ==================== DATA PREPROCESSING ====================
def preprocess_data(df):
    """Preprocess data - separate features and target."""
    # Separate features and target
    X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = df['species']
    
    # Clean species names (remove 'Iris-' prefix)
    y = y.str.replace('Iris-', '', regex=False)
    
    print("\nClass distribution:")
    print(y.value_counts())
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, X.columns

# ==================== TRAIN-TEST SPLIT ====================
def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into training and testing sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test

# ==================== MODEL TRAINING ===================
def train_models(X_train, y_train):
    """Train multiple classification models."""
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),
        'Support Vector Machine': SVC(
            kernel='rbf',
            C=1.0,
            random_state=42
        ),
        'K-Nearest Neighbors': KNeighborsClassifier(
            n_neighbors=5
        )
    }
    
    trained_models = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"{name} training completed!")
    
    return trained_models

# ==================== MODEL EVALUATION ====================
def evaluate_models(models, X_test, y_test):
    """Evaluate all trained models."""
    results = {}
    
    print("\n" + "="*70)
    print("MODEL EVALUATION RESULTS")
    print("="*70)
    
    for name, model in models.items():
        print(f"\n{name}:")
        print("-" * 50)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results[name] = {
            'model': model,
            'predictions': y_pred,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
    
    return results

# ==================== VISUALIZATION ====================
def plot_confusion_matrix(results, y_test):
    """Plot confusion matrices for all models."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Confusion Matrices', fontsize=16, fontweight='bold')
    
    for idx, (name, result) in enumerate(results.items()):
        cm = confusion_matrix(y_test, result['predictions'])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                      display_labels=['setosa', 'versicolor', 'virginica'])
        disp.plot(ax=axes[idx], cmap='Blues', values_format='d')
        axes[idx].set_title(f'{name}\nAccuracy: {result["accuracy"]:.4f}')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("\nConfusion matrices saved as 'confusion_matrices.png'")
    plt.show()

def plot_model_comparison(results):
    """Compare model performance."""
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    model_names = list(results.keys())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(metrics))
    width = 0.25
    
    for i, model_name in enumerate(model_names):
        values = [results[model_name][metric] for metric in metrics]
        ax.bar(x + i*width, values, width, label=model_name, alpha=0.8)
    
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim([0.95, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("Model comparison saved as 'model_comparison.png'")
    plt.show()

def plot_feature_importance(models):
    """Plot feature importance for Random Forest."""
    rf_model = models['Random Forest']
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    importance = rf_model.feature_importances_
    
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    bars = ax.barh(features, importance, color=colors)
    
    ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
    ax.set_title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("Feature importance plot saved as 'feature_importance.png'")
    plt.show()

# ==================== PREDICTION ON NEW DATA ====================
def predict_new_iris(model, scaler, measurements):
    """
    Predict iris species for new measurements.
    
    Parameters:
    -----------
    model : trained model
    scaler : StandardScaler object
    measurements : list or array [sepal_length, sepal_width, petal_length, petal_width]
    
    Returns:
    --------
    str : Predicted iris species
    """
    # Scale the input measurements
    measurements_scaled = scaler.transform([measurements])
    
    # Make prediction
    prediction = model.predict(measurements_scaled)[0]
    
    return prediction

# ==================== MAIN EXECUTION ====================
def main():
    """Main execution function."""
    print("="*70)
    print("IRIS FLOWER CLASSIFICATION")
    print("="*70)
    
    # 1. Load data
    print("\n[Step 1] Loading data...")
    df = load_data('IRIS.csv')
    
    # 2. Preprocess data
    print("\n[Step 2] Preprocessing data...")
    X_scaled, y, scaler, feature_names = preprocess_data(df)
    
    # 3. Split data
    print("\n[Step 3] Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)
    
    # 4. Train models
    print("\n[Step 4] Training models...")
    models = train_models(X_train, y_train)
    
    # 5. Evaluate models
    print("\n[Step 5] Evaluating models...")
    results = evaluate_models(models, X_test, y_test)
    
    # 6. Visualize results
    print("\n[Step 6] Creating visualizations...")
    plot_confusion_matrix(results, y_test)
    plot_model_comparison(results)
    plot_feature_importance(models)
    
    # 7. Find best model
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_model = results[best_model_name]['model']
    print(f"\n{'='*70}")
    print(f"BEST MODEL: {best_model_name}")
    print(f"Accuracy: {results[best_model_name]['accuracy']:.4f}")
    print(f"{'='*70}")
    
    # 8. Example predictions on new data
    print("\n[Step 7] Making predictions on new iris flowers...\n")
    
    # Example iris measurements
    examples = [
        ([5.1, 3.5, 1.4, 0.2], "Typical Setosa"),
        ([6.2, 2.9, 4.3, 1.3], "Typical Versicolor"),
        ([7.1, 3.0, 5.9, 2.1], "Typical Virginica"),
    ]
    
    for measurements, description in examples:
        prediction = predict_new_iris(best_model, scaler, measurements)
        print(f"{description}")
        print(f"  Measurements: {measurements}")
        print(f"  Predicted Species: {prediction}\n")
    
    print("="*70)
    print("CLASSIFICATION COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()
