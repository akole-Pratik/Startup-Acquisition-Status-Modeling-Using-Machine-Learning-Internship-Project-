# Startup-Acquisition-Status-Modeling-Using-Machine-Learning-Internship-Project-
Startup Acquisition Status Modeling Using Machine Learning 
# Building-Machine-Learning-Pipeline-on-Startups-Acquisition
## Aim: 
Our aim to understand the financial conditions of company fundraising goals.
## Description
The objective of the project is to predict whether a startup which is currently Operating, IPO, Acquired, or closed.This problem will be solved through a Supervised Machine Learning approach by training a model based on the history of startups which were either acquired or closed. The project aims to predict the acquisition status of startups based on various features such as funding rounds, total funding amount, industry category, and geographic location. By leveraging machine learning pipelines, we preprocess the data, select relevant features, and train models to classify startups into different acquisition status categories. The project utilizes Python libraries such as scikit-learn, pandas, matplotlib, seaborn, joblib, and XGBoost for model development and evaluation. The goal is to provide insights into the factors influencing startup acquisition and build a predictive tool that can assist stakeholders in making informed decisions.

## Understanding the Dataset
This project utilizes a dataset containing industry trends, investment insights, and company information.

Format: JSON and Excel
Link to Raw Data: Excel file
Columns: id, entity_type, name, category_code, status, founded_at, closed_at, domain, homepage_url, twitter_username, funding_total_usd, country_code, state_code, city, region, etc.
Data Information:
Total Records: 196,553
Data Columns: 44
Data Types: Object, Integer, Float
Missing Values: Present in multiple columns
This dataset serves as the foundation for building the machine learning model to predict the acquisition status of startups based on various features.
![image](https://github.com/akole-Pratik/Startup-Acquisition-Status-Modeling-Using-Machine-Learning-Internship-Project-/assets/81501071/fc07a2e1-107f-4973-8bac-517dfeb832cf)


##Data Preprocessing
The data preprocessing phase involved several steps, including:

Deleted columns providing excessive granularity such as 'region', 'city', 'state_code'
Removed redundant columns such as 'id', 'Unnamed: 0.1', 'entity_type'
Eliminated irrelevant features such as 'domain', 'homepage_url', 'twitter_username', 'logo_url'
Handled duplicate values
Removed columns with high null values
Dropped instances with missing values such as 'status', 'country_code', 'category_code', 'founded_at'
Dropped time-based columns such as 'first_investment_at', 'last_investment_at', 'first_funding_at'
Imputed missing values using mean() and mode() methods in numerical columns and categorical columns accordingly such as 'milestones', 'relationships', 'lat', 'lng'
After preprocessing, the DataFrame has the following information:

Total columns: 11
Non-Null Count: 63585
Data types: float64(7), object(4)
Memory usage: 7.8 MB
