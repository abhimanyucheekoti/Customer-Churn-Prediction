import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import shap

# Load dataset (adjust path if needed)
df = pd.read_csv(r"C:\Users\SHIVAKUMAR\Desktop\PythonPrograms\internship\Churn_Modelling.csv")

# Preview column names
print("Columns:", df.columns)

# Define the target and drop unnecessary columns
target = 'Exited'
drop_cols = ['RowNumber', 'CustomerId', 'Surname']  # not useful for prediction
X = df.drop(columns=[target] + drop_cols)
y = df[target]

# Identify feature types
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Preprocessing steps
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Modeling pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluation
print(classification_report(y_test, y_pred))
print("AUC Score:", roc_auc_score(y_test, y_prob))

# SHAP feature importance
explainer = shap.Explainer(model.named_steps['classifier'], feature_names=preprocessor.get_feature_names_out())
shap_values = explainer(model.named_steps['preprocessor'].transform(X_test))
shap.summary_plot(shap_values, features=model.named_steps['preprocessor'].transform(X_test),
                  feature_names=preprocessor.get_feature_names_out())
