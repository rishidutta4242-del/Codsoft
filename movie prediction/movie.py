import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Load data
df = pd.read_csv('IMDb Movies India.csv', encoding='latin-1')
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# Data cleaning
df = df.drop(['Name', 'Year'], axis=1)  # Drop non-predictive columns
df = df[df['Rating'].notna()].copy()  # Keep only rows with ratings

# Convert to numeric types
df['Duration'] = pd.to_numeric(df['Duration'].str.replace(' min', ''), errors='coerce')
df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')
df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')

# Drop rows with missing critical values
df = df.dropna(subset=['Duration', 'Rating'])

# Count genres (multiple genres separated by comma)
df['Genre_Count'] = df['Genre'].fillna('Unknown').str.split(',').str.len()

# Encode categorical variables
le_genre = LabelEncoder()
le_director = LabelEncoder()
df['Genre_Encoded'] = le_genre.fit_transform(df['Genre'].fillna('Unknown'))
df['Director_Encoded'] = le_director.fit_transform(df['Director'].fillna('Unknown'))

# Count actors (non-null actor columns)
df['Actor_Count'] = df[['Actor 1', 'Actor 2', 'Actor 3']].notna().sum(axis=1)

# Handle missing votes
df['Votes'] = df['Votes'].fillna(0)

# Select features and target
features = ['Duration', 'Genre_Encoded', 'Director_Encoded', 'Actor_Count', 'Votes', 'Genre_Count']
X = df[features].fillna(0)
y = df['Rating']

print(f"\n✓ Data prepared: {len(y)} samples with {len(features)} features")
print(f"Rating range: {y.min()} - {y.max()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

print("\n" + "="*60)
print("MODEL PERFORMANCE")
print("="*60)

best_model = None
best_r2 = -np.inf

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n{name}:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    
    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_name = name

print("\n" + "="*60)
print(f"✓ Best Model: {best_name} (R² = {best_r2:.4f})")
print("="*60)

# Feature importance
print("\nFeature Importance:")
importances = best_model.feature_importances_
for feat, imp in sorted(zip(features, importances), key=lambda x: x[1], reverse=True):
    print(f"  {feat}: {imp:.4f}")

# Predictions on sample
print("\nSample Predictions (Test Set):")
sample_indices = np.random.choice(len(y_test), 5, replace=False)
for idx in sample_indices:
    pred = best_model.predict(X_test.iloc[[idx]])[0]
    actual = y_test.iloc[idx]
    print(f"  Predicted: {pred:.2f} | Actual: {actual:.2f}")

print("\n✓ Model training complete!")

# VISUALIZATIONS
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

fig = plt.figure(figsize=(16, 12))

# 1. Feature Importance
ax1 = plt.subplot(3, 3, 1)
feat_imp_sorted = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
feat_names, imp_vals = zip(*feat_imp_sorted)
ax1.barh(feat_names, imp_vals, color='steelblue')
ax1.set_xlabel('Importance', fontsize=10)
ax1.set_title('Feature Importance', fontsize=12, fontweight='bold')
ax1.invert_yaxis()

# 2. Rating Distribution
ax2 = plt.subplot(3, 3, 2)
ax2.hist(y, bins=30, color='coral', edgecolor='black', alpha=0.7)
ax2.set_xlabel('Rating', fontsize=10)
ax2.set_ylabel('Frequency', fontsize=10)
ax2.set_title('Movie Rating Distribution', fontsize=12, fontweight='bold')
ax2.grid(alpha=0.3)

# 3. Actual vs Predicted (Test Set)
ax3 = plt.subplot(3, 3, 3)
ax3.scatter(y_test, y_pred_test, alpha=0.5, s=30, color='green')
ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Fit')
ax3.set_xlabel('Actual Rating', fontsize=10)
ax3.set_ylabel('Predicted Rating', fontsize=10)
ax3.set_title('Actual vs Predicted (Test Set)', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3)

# 4. Residuals Plot
ax4 = plt.subplot(3, 3, 4)
residuals = y_test - y_pred_test
ax4.scatter(y_pred_test, residuals, alpha=0.5, s=30, color='purple')
ax4.axhline(y=0, color='r', linestyle='--', lw=2)
ax4.set_xlabel('Predicted Rating', fontsize=10)
ax4.set_ylabel('Residuals', fontsize=10)
ax4.set_title('Residuals Plot', fontsize=12, fontweight='bold')
ax4.grid(alpha=0.3)

# 5. Model Comparison - R² Score
ax5 = plt.subplot(3, 3, 5)
model_names = list(models.keys())
r2_scores = []
for name, model in models.items():
    pred = model.predict(X_test)
    r2_scores.append(r2_score(y_test, pred))
colors = ['steelblue' if r2 == best_r2 else 'lightblue' for r2 in r2_scores]
ax5.bar(model_names, r2_scores, color=colors, edgecolor='black')
ax5.set_ylabel('R² Score', fontsize=10)
ax5.set_title('Model Comparison - R² Score', fontsize=12, fontweight='bold')
ax5.grid(alpha=0.3, axis='y')

# 6. Model Comparison - MAE
ax6 = plt.subplot(3, 3, 6)
mae_scores = []
for name, model in models.items():
    pred = model.predict(X_test)
    mae_scores.append(mean_absolute_error(y_test, pred))
ax6.bar(model_names, mae_scores, color='lightcoral', edgecolor='black')
ax6.set_ylabel('MAE', fontsize=10)
ax6.set_title('Model Comparison - MAE', fontsize=12, fontweight='bold')
ax6.grid(alpha=0.3, axis='y')

# 7. Prediction Error Distribution
ax7 = plt.subplot(3, 3, 7)
errors = np.abs(y_test - y_pred_test)
ax7.hist(errors, bins=30, color='orange', edgecolor='black', alpha=0.7)
ax7.set_xlabel('Absolute Error', fontsize=10)
ax7.set_ylabel('Frequency', fontsize=10)
ax7.set_title('Prediction Error Distribution', fontsize=12, fontweight='bold')
ax7.grid(alpha=0.3)

# 8. Train vs Test Performance
ax8 = plt.subplot(3, 3, 8)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
metrics = ['R² Score', 'RMSE', 'MAE']
train_vals = [train_r2, np.sqrt(mean_squared_error(y_train, y_pred_train)), mean_absolute_error(y_train, y_pred_train)]
test_vals = [test_r2, np.sqrt(mean_squared_error(y_test, y_pred_test)), mean_absolute_error(y_test, y_pred_test)]
x = np.arange(len(metrics))
width = 0.35
ax8.bar(x - width/2, train_vals, width, label='Train', color='skyblue', edgecolor='black')
ax8.bar(x + width/2, test_vals, width, label='Test', color='salmon', edgecolor='black')
ax8.set_ylabel('Score', fontsize=10)
ax8.set_title('Train vs Test Performance', fontsize=12, fontweight='bold')
ax8.set_xticks(x)
ax8.set_xticklabels(metrics)
ax8.legend()
ax8.grid(alpha=0.3, axis='y')

# 9. Prediction Range Analysis
ax9 = plt.subplot(3, 3, 9)
score_range = np.linspace(y_pred_test.min(), y_pred_test.max(), 6)
for i in range(len(score_range)-1):
    mask = (y_pred_test >= score_range[i]) & (y_pred_test < score_range[i+1])
    ax9.scatter([i]*mask.sum(), y_test[mask], alpha=0.5, s=20)
ax9.set_xticks(range(len(score_range)-1))
ax9.set_xticklabels([f'{score_range[i]:.1f}' for i in range(len(score_range)-1)], fontsize=8)
ax9.set_xlabel('Predicted Rating Range', fontsize=10)
ax9.set_ylabel('Actual Rating', fontsize=10)
ax9.set_title('Actual Ratings by Prediction Range', fontsize=12, fontweight='bold')
ax9.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('movie_rating_predictions.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved as 'movie_rating_predictions.png'")
plt.show()
