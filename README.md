# 📊 Customer Churn Decision Intelligence Platform

An end-to-end, explainable data decision platform that analyzes customer behavior, predicts churn risk, explains *why* churn happens, and supports actionable business decisions.

This project mirrors how real companies use data: from raw ingestion → modeling → explainability → deployment.

---

## 🔍 Problem Statement

Customer churn is expensive. Retaining existing customers is significantly cheaper than acquiring new ones.

**Business question:**
> Given historical customer data, who is likely to churn, why are they at risk, and what actions should decision-makers take?

---

## 🧠 Solution Overview

This project delivers a **Decision Intelligence System** that answers:

1. What is happening  
2. Why it is happening  
3. What will happen next  
4. What should decision-makers do  

---

## 🗂️ Project Structure

- data/ — raw and processed data  
- src/ — pipelines and models  
- app/ — Streamlit application  
- reports/ — figures and models  

---

## 📊 Dataset

- IBM Telco Customer Churn
- 7,043 customers
- Target: churn (1 = churned, 0 = retained)

---

## ⚙️ Modeling & Explainability

- Logistic Regression (ROC-AUC: 0.846)
- SHAP for global feature importance

Key drivers:
- Short tenure
- Month-to-month contracts
- Higher monthly charges

---

## 🧪 Interactive App

A Streamlit app allows users to simulate customer profiles and receive churn risk estimates with business recommendations.

---

## 🛠 Tech Stack

Python, Pandas, NumPy, Scikit-learn, SHAP, Streamlit

---

## 🎯 Why This Matters

This project demonstrates end-to-end analytics, explainable ML, and deployment-ready thinking.

