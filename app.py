import streamlit as st
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report)

# --- App Config ---
st.set_page_config(page_title="BITS Assignment: Bank Churn Predictor", layout="wide")


# --- Asset Loading ---
@st.cache_resource
def load_model_assets():
    models = {}
    # Expected folder structure: models/logistic_regression.pkl, etc.
    model_names = ["logistic_regression", "decision_tree", "knn", "naive_bayes", "random_forest", "xgboost"]
    for m in model_names:
        try:
            with open(f"models/{m}.pkl", "rb") as f:
                models[m.replace("_", " ").title()] = pickle.load(f)
        except FileNotFoundError:
            st.error(
                f"Model file 'models/{m}.pkl' not found. Please ensure the 'models' folder exists in your repository.")

    try:
        with open("models/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        scaler = None
        st.warning("Scaler file 'models/scaler.pkl' not found. Distance-based models may not function correctly.")

    return models, scaler


# --- UI Header ---
st.title("🏦 Bank Churn Prediction Dashboard")
st.markdown("""
### **M.Tech Data Science & Engineering | BITS Pilani**
*Assignment 2 by **Sobana S***
""")

models, scaler = load_model_assets()

# --- Sidebar Controls ---
st.sidebar.header("Control Panel")
app_mode = st.sidebar.selectbox("Choose Mode", ["Batch Evaluation (CSV)", "Single Customer Prediction"])

if models:
    selected_model_name = st.sidebar.selectbox("Select ML Model Engine", list(models.keys()), index=len(models) - 1)
    model = models[selected_model_name]
else:
    st.stop()


# --- Helper Function: Preprocessing ---
def preprocess_data(df):
    # 1. Drop irrelevant identifiers
    cols_to_drop = ['RowNumber', 'CustomerId', 'Surname']
    df_proc = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # 2. Encode Categorical variables
    geo_map = {"France": 0, "Germany": 1, "Spain": 2}
    gender_map = {"Female": 0, "Male": 1}

    if 'Geography' in df_proc.columns and df_proc['Geography'].dtype == 'O':
        df_proc['Geography'] = df_proc['Geography'].map(geo_map)
    if 'Gender' in df_proc.columns and df_proc['Gender'].dtype == 'O':
        df_proc['Gender'] = df_proc['Gender'].map(gender_map)

    # 3. Feature Engineering
    if 'Balance_Salary_Ratio' not in df_proc.columns:
        df_proc['Balance_Salary_Ratio'] = df_proc['Balance'] / (df_proc['EstimatedSalary'] + 1)
    if 'Tenure_By_Age' not in df_proc.columns:
        df_proc['Tenure_By_Age'] = df_proc['Tenure'] / (df_proc['Age'] + 1)
    if 'Is_Senior' not in df_proc.columns:
        df_proc['Is_Senior'] = df_proc['Age'].apply(lambda x: 1 if x >= 60 else 0)

    # 4. Align Columns with Training
    expected_cols = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
                     'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
                     'Balance_Salary_Ratio', 'Tenure_By_Age', 'Is_Senior']

    # Fill any missing values with 0 to prevent crash
    return df_proc[expected_cols].fillna(0)


# --- MODE 1: BATCH EVALUATION ---
if app_mode == "Batch Evaluation (CSV)":
    st.subheader("📁 Batch Performance Evaluation")
    uploaded_file = st.file_uploader("Upload Test CSV (Must contain 'Exited' column)", type="csv")

    if uploaded_file:
        test_df = pd.read_csv(uploaded_file)
        if 'Exited' not in test_df.columns:
            st.error("Error: Target column 'Exited' missing from CSV.")
        else:
            y_test = test_df['Exited']
            X_test_proc = preprocess_data(test_df)

            # Scale if needed for specific models
            needs_scaling = any(x in selected_model_name for x in ["Logistic", "Knn", "Naive"])
            X_eval = scaler.transform(X_test_proc) if (scaler and needs_scaling) else X_test_proc

            # Predictions
            y_pred = model.predict(X_eval)
            y_prob = model.predict_proba(X_eval)[:, 1] if hasattr(model, "predict_proba") else y_pred

            # Display metrics
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
            m2.metric("AUC", f"{roc_auc_score(y_test, y_prob):.3f}")
            m3.metric("Precision", f"{precision_score(y_test, y_pred):.3f}")
            m4.metric("Recall", f"{recall_score(y_test, y_pred):.3f}")
            m5.metric("F1", f"{f1_score(y_test, y_pred):.3f}")
            m6.metric("MCC", f"{matthews_corrcoef(y_test, y_pred):.3f}")

            # --- Visual Insights ---
            st.divider()
            st.subheader("📊 Data Exploration & Banking Insights")

            v_col1, v_col2 = st.columns(2)

            with v_col1:
                st.write("**Balance vs. Age (Colored by Churn)**")
                fig, ax = plt.subplots()
                sns.scatterplot(data=test_df, x='Age', y='Balance', hue='Exited', alpha=0.6, palette='viridis', ax=ax)
                plt.title("Wealth Concentration by Age Group")
                st.pyplot(fig)
                st.info(
                    "💡 Insight: High-balance churners often cluster in the 45-60 age band, indicating potential 'wealth-exit' patterns.")

            with v_col2:
                st.write("**Churn Rate by Geography**")
                fig, ax = plt.subplots()
                geo_churn = test_df.groupby('Geography')['Exited'].mean() * 100
                geo_churn.plot(kind='bar', color=['#004a99', '#ffcc00', '#c60c30'], ax=ax)
                ax.set_ylabel("Churn Percentage (%)")
                st.pyplot(fig)
                st.info("💡 Insight: Significant regional variance helps prioritize localized retention campaigns.")

            st.divider()
            # Confusion Matrix & Classification Report
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Confusion Matrix**")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                            xticklabels=['Stayed (0)', 'Churned (1)'],
                            yticklabels=['Stayed (0)', 'Churned (1)'])
                st.pyplot(fig)
            with c2:
                st.write("**Classification Report**")
                report = classification_report(y_test, y_pred,
                                               target_names=['Stayed (0)', 'Churned (1)'],
                                               output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose().style.background_gradient(cmap='Greens'))

# --- MODE 2: SINGLE PREDICTION ---
else:
    st.subheader("👤 Individual Customer Risk Analysis")
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 95, 35)
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Female", "Male"])
        credit_score = st.number_input("Credit Score", 300, 850, 650)
        tenure = st.slider("Tenure (Years)", 0, 10, 5)

    with col2:
        balance = st.number_input("Balance ($)", 0.0, 250000.0, 50000.0)
        num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
        has_card = st.toggle("Has Credit Card", value=True)
        is_active = st.toggle("Is Active Member", value=True)
        salary = st.number_input("Estimated Salary ($)", 0.0, 250000.0, 75000.0)

    if st.button("Analyze Risk"):
        # Build single row DataFrame
        single_df = pd.DataFrame([{
            'CreditScore': credit_score, 'Geography': geography, 'Gender': gender,
            'Age': age, 'Tenure': tenure, 'Balance': balance, 'NumOfProducts': num_products,
            'HasCrCard': int(has_card), 'IsActiveMember': int(is_active), 'EstimatedSalary': salary
        }])

        X_single = preprocess_data(single_df)

        # Scale if required
        needs_scaling = any(x in selected_model_name for x in ["Logistic", "Knn", "Naive"])
        X_final = scaler.transform(X_single) if (scaler and needs_scaling) else X_single

        # Prediction probability
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_final)[0][1]
        else:
            prob = float(model.predict(X_final)[0])

        st.divider()
        st.metric("Churn Probability", f"{prob:.1%}")

        if prob > 0.5:
            st.error(f"🚨 **HIGH RISK**: Customer is likely to CHURN ({prob:.1%} probability)")
            st.write("---")
            st.warning("Recommendation: Proactive outreach with loyalty offers or fee waivers is advised.")
        else:
            st.success(f"✅ **LOW RISK**: Customer is likely to STAY ({prob:.1%} churn probability)")
            st.write("---")
            st.info(
                "Recommendation: Maintain standard communication; high potential for cross-selling additional products.")

st.sidebar.markdown("---")
st.sidebar.write("Developed by **Sobana S**")