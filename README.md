a. Problem Statement

The objective of this project is to predict bank customer churn (the likelihood of a customer leaving the bank) using various machine learning classification models. In the banking sector, identifying at-risk customers is crucial for implementing proactive retention strategies and maintaining a healthy Lifetime Value (LTV) for the customer base.

b. Dataset Description

Source: Bank Churn Modelling dataset (Kaggle).

Instance Size: 10,000 records.

Target Variable: Exited (1 = Churned, 0 = Stayed).

Features Used (13 total): * Demographics: Age, Geography, Gender.

Financials: Credit Score, Balance, Estimated Salary.

Engagement: Tenure, Number of Products, Has CrCard, IsActiveMember.

Engineered: Balance_Salary_Ratio, Tenure_By_Age, Is_Senior.

c. Models Used & Comparison Table

Six models were implemented on the same dataset with a 20% test split.
--------------------------------------------------------------------
ML Model Name     	Accuracy	AUC	Precision	Recall	F1     	MCC
---------------------------------------------------------------------
Logistic Regression	0.8240	0.7878	0.6897	0.2457	0.3623	0.3376
Decision Tree	      0.8555	0.8389	0.8138	0.3759	0.5143	0.4883
kNN               	0.8350	0.7728	0.6711	0.3710	0.4778	0.4135
Naive Bayes	        0.7925	0.7398	0.1667	0.0049	0.0095	-0.0071
Random Forest	      0.8590	0.8425	0.7854	0.4226	0.5495	0.5068
XGBoost	            0.8455	0.8293	0.6655	0.4840	0.5605	0.4783
-------------------------------------------------------------------
