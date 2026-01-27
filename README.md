Bank Customer Churn Prediction
Project Overview
This project aims to predict the likelihood of bank customers leaving the institution (churning). By analyzing demographic and financial data, the model identifies high-risk customers, allowing the bank to take proactive retention measures.

Dataset
The project uses the Churn_Modelling.csv dataset, which consists of 10,000 customer records with 14 features, including:

Credit Score, Age, Tenure, Balance

Number of Products, Credit Card Status, Active Membership

Estimated Salary and Geography

Machine Learning Models
I implemented and compared six classification algorithms to determine the best performer:

Random Forest (Best Performer: ~85.9% Accuracy, 0.842 AUC)

XGBoost

Logistic Regression

Decision Tree

K-Nearest Neighbor (KNN)

Naive Bayes

Key Technical Implementations
Exploratory Data Analysis (EDA): Visualizing distributions and correlations to understand churn drivers.

Scalable Processing: Developed a version of the pipeline using PySpark on Databricks for handling larger datasets.

Model Evaluation: Focused on Accuracy and AUC-ROC scores to ensure balanced performance.

Deployment: The final model is deployed as a real-time risk analysis tool using Streamlit Community Cloud.

Installation & Usage

Clone the repository:

Bash
git clone https://github.com/yourusername/bank-churn-prediction.git

Install requirements:

Bash
pip install -r requirements.txt
Run the application:

Bash
streamlit run streamlit_app.py

Conclusion
The Random Forest model provides the most reliable predictions for this dataset. This tool serves as a decision-support system for relationship managers to focus their efforts on customers most likely to exit.

