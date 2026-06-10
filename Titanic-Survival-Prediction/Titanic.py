"""
Titanic Survival Prediction Model
This script builds machine learning models to predict passenger survival on the Titanic.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def load_and_explore_data(filepath):
    """Load and display basic information about the dataset."""
    print("="*70)
    print("LOADING AND EXPLORING DATASET")
    print("="*70)
    
    df = pd.read_csv(filepath)
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nDataset Info:")
    print(df.info())
    print(f"\nStatistical Summary:")
    print(df.describe())
    print(f"\nMissing Values:")
    print(df.isnull().sum())
    
    return df


def preprocess_data(df):
    """Handle missing values, duplicates, and data inconsistencies."""
    print("\n" + "="*70)
    print("DATA PREPROCESSING AND CLEANING")
    print("="*70)
    
    df_processed = df.copy()
    
    # Remove duplicates
    df_processed = df_processed.drop_duplicates()
    print(f"\nDuplicate rows removed. New shape: {df_processed.shape}")
    
    # Handle missing values
    print("\nHandling missing values:")
    
    # Age: Fill with median
    if 'Age' in df_processed.columns:
        df_processed['Age'].fillna(df_processed['Age'].median(), inplace=True)
        print("  Age: Filled with median")
    
    # Embarked: Fill with mode
    if 'Embarked' in df_processed.columns:
        df_processed['Embarked'].fillna(df_processed['Embarked'].mode()[0], inplace=True)
        print("  Embarked: Filled with mode")
    
    # Fare: Fill with median
    if 'Fare' in df_processed.columns:
        df_processed['Fare'].fillna(df_processed['Fare'].median(), inplace=True)
        print("  Fare: Filled with median")
    
    # Cabin: Too many missing values, will be dropped
    if 'Cabin' in df_processed.columns:
        df_processed.drop('Cabin', axis=1, inplace=True)
        print("  Cabin: Dropped (too many missing values)")
    
    print(f"\nMissing values after preprocessing:")
    print(df_processed.isnull().sum())
    
    return df_processed


def feature_engineering(df_processed):
    """Create new features and encode categorical variables."""
    print("\n" + "="*70)
    print("FEATURE ENGINEERING")
    print("="*70)
    
    # 1. Extract title from Name
    if 'Name' in df_processed.columns:
        df_processed['Title'] = df_processed['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        print(f"Title values: {df_processed['Title'].unique()}")
    
    # 2. Family size features
    if 'SibSp' in df_processed.columns and 'Parch' in df_processed.columns:
        df_processed['FamilySize'] = df_processed['SibSp'] + df_processed['Parch'] + 1
        df_processed['IsAlone'] = (df_processed['FamilySize'] == 1).astype(int)
        print("Family size features created")
    
    # 3. Age groups
    if 'Age' in df_processed.columns:
        df_processed['AgeGroup'] = pd.cut(df_processed['Age'], bins=[0, 12, 18, 35, 60, 100], 
                                          labels=['Child', 'Teen', 'Adult', 'Middle-aged', 'Senior'])
        print("Age groups created")
    
    # 4. Fare per person
    if 'Fare' in df_processed.columns and 'FamilySize' in df_processed.columns:
        df_processed['FarePerPerson'] = df_processed['Fare'] / df_processed['FamilySize']
        print("Fare per person calculated")
    
    print("\nNew features created successfully!")
    
    # Encode categorical variables
    print("\nEncoding categorical variables:")
    
    # Sex encoding
    if 'Sex' in df_processed.columns:
        df_processed['Sex'] = df_processed['Sex'].map({'male': 1, 'female': 0})
        print("  Sex: Encoded (male=1, female=0)")
    
    # Embarked encoding
    if 'Embarked' in df_processed.columns:
        embarked_mapping = {'S': 0, 'C': 1, 'Q': 2}
        df_processed['Embarked'] = df_processed['Embarked'].map(embarked_mapping)
        print("  Embarked: Encoded (S=0, C=1, Q=2)")
    
    # Title encoding
    if 'Title' in df_processed.columns:
        title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Other': 0}
        df_processed['Title'] = df_processed['Title'].map(title_mapping)
        df_processed['Title'].fillna(0, inplace=True)
        print("  Title: Encoded")
    
    # AgeGroup encoding
    if 'AgeGroup' in df_processed.columns:
        df_processed['AgeGroup'] = pd.factorize(df_processed['AgeGroup'])[0]
        print("  AgeGroup: Encoded")
    
    print("\nFeature engineering completed!")
    
    return df_processed


def prepare_features(df_processed):
    """Select and prepare features for modeling."""
    print("\n" + "="*70)
    print("PREPARING FEATURES")
    print("="*70)
    
    features_to_use = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 
                       'Title', 'FamilySize', 'IsAlone', 'FarePerPerson', 'AgeGroup']
    
    # Keep only available features
    available_features = [f for f in features_to_use if f in df_processed.columns]
    print(f"Features to use: {available_features}")
    
    # Prepare X and y
    X = df_processed[available_features].copy()
    y = df_processed['Survived'].copy()
    
    print(f"\nFilling NaN values using SimpleImputer...")
    print(f"NaN values before imputation:\n{X.isnull().sum()}")
    
    # Use SimpleImputer to handle all NaN values with median strategy
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=available_features)
    
    print(f"NaN values after imputation:\n{X.isnull().sum()}")
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target variable shape: {y.shape}")
    
    return X, y, available_features


def split_and_scale_data(X, y):
    """Split data into training and testing sets, and scale features."""
    print("\n" + "="*70)
    print("SPLITTING AND SCALING DATA")
    print("="*70)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                          random_state=42, stratify=y)
    
    print(f"Training set size: {X_train.shape[0]} ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"Testing set size: {X_test.shape[0]} ({X_test.shape[0]/len(X)*100:.1f}%)")
    
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nFeatures scaled successfully!")
    print(f"Training set scaled shape: {X_train_scaled.shape}")
    print(f"Testing set scaled shape: {X_test_scaled.shape}")
    
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_models(X_train, X_test, X_train_scaled, X_test_scaled, y_train):
    """Train multiple classification models."""
    print("\n" + "="*70)
    print("TRAINING MODELS")
    print("="*70)
    
    models = {}
    predictions = {}
    
    # 1. Logistic Regression
    print("\nTraining Logistic Regression...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    models['Logistic Regression'] = lr_model
    predictions['Logistic Regression'] = lr_model.predict(X_test_scaled)
    print("✓ Logistic Regression trained!")
    
    # 2. Random Forest
    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    models['Random Forest'] = rf_model
    predictions['Random Forest'] = rf_model.predict(X_test)
    print("✓ Random Forest trained!")
    
    # 3. Gradient Boosting
    print("\nTraining Gradient Boosting...")
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)
    models['Gradient Boosting'] = gb_model
    predictions['Gradient Boosting'] = gb_model.predict(X_test)
    print("✓ Gradient Boosting trained!")
    
    print("\n" + "="*50)
    print("All models trained successfully!")
    print("="*50)
    
    return models, predictions


def evaluate_models(predictions, y_test):
    """Evaluate all models and display results."""
    print("\n" + "="*70)
    print("MODEL EVALUATION RESULTS")
    print("="*70)
    
    results = []
    
    for model_name, y_pred in predictions.items():
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })
        
        print(f"\n{model_name}:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    print("\n" + "="*70)
    print("Summary Table:")
    print(results_df.to_string(index=False))
    print("="*70)
    
    return results_df


def visualize_confusion_matrices(predictions, y_test):
    """Visualize confusion matrices for all models."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, (model_name, y_pred) in enumerate(predictions.items()):
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], 
                    xticklabels=['Died', 'Survived'], 
                    yticklabels=['Died', 'Survived'],
                    cbar=False)
        axes[idx].set_title(f'{model_name}\nConfusion Matrix')
        axes[idx].set_ylabel('Actual')
        axes[idx].set_xlabel('Predicted')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=100, bbox_inches='tight')
    print("\nConfusion matrices saved as 'confusion_matrices.png'")
    plt.show()


def print_classification_reports(predictions, y_test):
    """Print detailed classification reports."""
    print("\n" + "="*70)
    print("DETAILED CLASSIFICATION REPORTS")
    print("="*70)
    
    for model_name, y_pred in predictions.items():
        print(f"\n{model_name}:")
        print(classification_report(y_test, y_pred, target_names=['Died', 'Survived']))


def display_sample_predictions(models, X_test, y_test):
    """Display sample predictions on test set."""
    print("\n" + "="*70)
    print("SAMPLE PREDICTIONS ON TEST SET")
    print("="*70)
    
    best_model = models['Random Forest']
    
    # Display predictions for first 10 test samples
    for i in range(min(10, len(X_test))):
        sample_data = X_test.iloc[i].values
        actual_survival = y_test.iloc[i]
        
        prediction = best_model.predict([sample_data])[0]
        probability = best_model.predict_proba([sample_data])[0]
        
        print(f"\nPassenger {i+1}:")
        print(f"  Actual:     {'Survived' if actual_survival == 1 else 'Died'}")
        print(f"  Predicted:  {'Survived' if prediction == 1 else 'Died'}")
        print(f"  Confidence: {max(probability)*100:.2f}%")


def analyze_feature_importance(models, available_features):
    """Analyze and visualize feature importance."""
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    # Feature Importance for Random Forest
    rf_importance = models['Random Forest'].feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': available_features,
        'Importance': rf_importance
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance (Random Forest):")
    print(feature_importance_df.to_string(index=False))
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='steelblue')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.title('Feature Importance for Titanic Survival Prediction')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=100, bbox_inches='tight')
    print("\nFeature importance plot saved as 'feature_importance.png'")
    plt.show()


def main():
    """Main execution function."""
    print("\n")
    print("*" * 70)
    print("TITANIC SURVIVAL PREDICTION MODEL")
    print("*" * 70)
    
    try:
        # Load and explore data
        df = load_and_explore_data('Titanic-Dataset.csv')
        
        # Preprocess data
        df_processed = preprocess_data(df)
        
        # Feature engineering
        df_processed = feature_engineering(df_processed)
        
        # Prepare features
        X, y, available_features = prepare_features(df_processed)
        
        # Split and scale data
        X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale_data(X, y)
        
        # Train models
        models, predictions = train_models(X_train, X_test, X_train_scaled, X_test_scaled, y_train)
        
        # Evaluate models
        results_df = evaluate_models(predictions, y_test)
        
        # Visualize confusion matrices
        visualize_confusion_matrices(predictions, y_test)
        
        # Print classification reports
        print_classification_reports(predictions, y_test)
        
        # Display sample predictions
        display_sample_predictions(models, X_test, y_test)
        
        # Analyze feature importance
        analyze_feature_importance(models, available_features)
        
        print("\n" + "="*70)
        print("NOTEBOOK EXECUTION COMPLETE!")
        print("="*70 + "\n")
        
    except FileNotFoundError:
        print("Error: 'Titanic-Dataset.csv' not found. Please ensure the CSV file is in the same directory.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
