import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef, confusion_matrix)


# ==========================================
# 1. DATA LOADING & CLEANING
# ==========================================
def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    # Drop irrelevant identifiers for modeling (Step 1: Dataset choice)
    # Bank Churn dataset has 14 features, we use 11-13 relevant ones.
    df_clean = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

    # Check for missing values
    if df_clean.isnull().sum().sum() > 0:
        df_clean = df_clean.dropna()

    return df_clean


# ==========================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
def perform_eda(df):
    print("--- Basic Statistics ---")
    print(df.describe())

    # Correlation Heatmap
    plt.figure(figsize=(10, 6))
    temp_df = df.copy()
    for col in temp_df.select_dtypes('object').columns:
        temp_df[col] = LabelEncoder().fit_transform(temp_df[col])
    sns.heatmap(temp_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Feature Correlation Heatmap")
    plt.savefig('correlation_heatmap.png')
    print("EDA Visuals saved: correlation_heatmap.png")


# ==========================================
# 3. FEATURE ENGINEERING & FEATURE STORE
# ==========================================
def feature_engineering(df):
    # Derived Features
    df['Balance_Salary_Ratio'] = df['Balance'] / (df['EstimatedSalary'] + 1)
    df['Tenure_By_Age'] = df['Tenure'] / df['Age']
    df['Is_Senior'] = df['Age'].apply(lambda x: 1 if x >= 60 else 0)

    # Encoding Categorical Features
    le_geo = LabelEncoder()
    le_gender = LabelEncoder()
    df['Geography'] = le_geo.fit_transform(df['Geography'])
    df['Gender'] = le_gender.fit_transform(df['Gender'])

    # Save encoders for Streamlit
    if not os.path.exists('models'): os.makedirs('models')
    with open('models/le_geo.pkl', 'wb') as f:
        pickle.dump(le_geo, f)
    with open('models/le_gender.pkl', 'wb') as f:
        pickle.dump(le_gender, f)

    # Simulation: Save engineered features
    if not os.path.exists('feature_store'):
        os.makedirs('feature_store')
    df.to_parquet('feature_store/bank_churn_features.parquet')

    return df


# ==========================================
# 4. MODEL TRAINING & EVALUATION (Step 2)
# ==========================================
def train_and_evaluate(df):
    X = df.drop('Exited', axis=1)
    y = df['Exited']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Save test data for Streamlit upload requirement (Step 6a)
    X_test_with_target = X_test.copy()
    X_test_with_target['Exited'] = y_test
    X_test_with_target.to_csv('test_data.csv', index=False)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Required 6 ML Models
    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(max_depth=5),
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(eval_metric='logloss')
    }

    results = []

    for name, model in models.items():
        # Use scaled data for distance-based models
        X_tr = X_train_scaled if name in ["Logistic Regression", "kNN"] else X_train
        X_ts = X_test_scaled if name in ["Logistic Regression", "kNN"] else X_test

        model.fit(X_tr, y_train)
        y_pred = model.predict(X_ts)
        y_prob = model.predict_proba(X_ts)[:, 1] if hasattr(model, "predict_proba") else y_pred

        # Required 6 Evaluation Metrics
        metrics = {
            "ML Model Name": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "AUC": roc_auc_score(y_test, y_prob),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "MCC": matthews_corrcoef(y_test, y_pred)
        }
        results.append(metrics)

        # Save models (Step 3)
        model_path = f'models/{name.lower().replace(" ", "_")}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

    # Save Scaler
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    return pd.DataFrame(results)


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    if not os.path.exists('Churn_Modelling.csv'):
        print("Error: Churn_Modelling.csv not found.")
    else:
        raw_data = load_and_clean_data('Churn_Modelling.csv')
        perform_eda(raw_data)
        engineered_data = feature_engineering(raw_data)
        comparison_table = train_and_evaluate(engineered_data)

        print("\n--- FINAL MODEL COMPARISON ---")
        print(comparison_table.to_string(index=False))
        comparison_table.to_csv('model_comparison_results.csv', index=False)