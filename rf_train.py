
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# =====================
# Load dataset
# =====================
df = pd.read_csv('https://raw.githubusercontent.com/Abhishek20182/mobile-price-classification/main/train.csv')

print("Dataset loaded successfully!")
print(df.head())

# =====================
# Feature Engineering
# =====================
df['total_camera'] = df['pc'] + df['fc']
df['screen_area'] = df['sc_h'] * df['sc_w']
df['pixel_area'] = df['px_height'] * df['px_width']
df['battery_per_weight'] = df['battery_power'] / (df['mobile_wt'] + 1)
df['premium_features'] = df['blue'] + df['dual_sim'] + df['four_g'] + df['wifi']

print("\n✅ Created 5 engineered features")

# Target and features
X = df.drop('price_range', axis=1)
y = df['price_range']

# =====================
# Column split
# =====================
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

print(f"\nNumeric features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")

# =====================
# Preprocessing
# =====================
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, numeric_features),
    ('cat', cat_transformer, categorical_features)
])

# =====================
# Random Forest Model
# =====================
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)

# =====================
# Full Pipeline
# =====================
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', rf_model)
])

# =====================
# Train-test split
# ====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# =====================
# Training
# =====================
print("\n🚀 Training model...")
rf_pipeline.fit(X_train, y_train)

# =====================
# Evaluation
# =====================
y_pred = rf_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'='*60}")
print(f"Accuracy: {accuracy:.4f}")
print(f"{'='*60}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, 
                          target_names=['Low', 'Medium', 'High', 'Very High']))

# =====================
# Save model (IMPORTANT)
# =====================

with open("mobile_price_pipeline.pkl", "wb") as f:
    pickle.dump(rf_pipeline, f)

print("\n✅ Random Forest pipeline saved as mobile_price_pipeline.pkl")
