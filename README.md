# Startup-Acquisition-Status-Modeling-Using-Machine-Learning-Internship-Project-
Startup Acquisition Status Modeling Using Machine Learning 
# Building-Machine-Learning-Pipeline-on-Startups-Acquisition
## Aim: 
Our aim to understand the financial conditions of company fundraising goals.
##Description
The objective of the project is to predict whether a startup which is currently Operating, IPO, Acquired, or closed.This problem will be solved through a Supervised Machine Learning approach by training a model based on the history of startups which were either acquired or closed. The project aims to predict the acquisition status of startups based on various features such as funding rounds, total funding amount, industry category, and geographic location. By leveraging machine learning pipelines, we preprocess the data, select relevant features, and train models to classify startups into different acquisition status categories. The project utilizes Python libraries such as scikit-learn, pandas, matplotlib, seaborn, joblib, and XGBoost for model development and evaluation. The goal is to provide insights into the factors influencing startup acquisition and build a predictive tool that can assist stakeholders in making informed decisions.

##Understanding the Dataset
This project utilizes a dataset containing industry trends, investment insights, and company information.

Format: Excel

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

#Data Cleaning
Data cleaning is a crucial step in the data preprocessing pipeline to ensure the quality and reliability of the dataset. This section outlines the steps taken to clean the startup dataset.

1. Delete Irrelevant and Redundant Information
    a. Delete Granular Features
region, city, state_code: These features provide excessive granularity and are not directly relevant to the target variables.
    b. Delete Redundant Features
id, Unnamed: 0.1, entity_type, entity_id, parent_id, created_by, created_at, updated_at: These features are redundant and do not contribute to the prediction model.
    c. Delete Irrelevant Features
domain, homepage_url, twitter_username, logo_url, logo_width, logo_height, short_description, description, overview, tag_list, name, normalized_name, permalink, invested_companies: These features are not relevant to the prediction task.
    d. Delete Duplicate Values
Removed any duplicate rows in the dataset to avoid bias in the model.
    e. Delete Null Values
Removed features with more than 98% null values to ensure data quality.
2. Remove Noise or Unreliable Data
    a. Delete Instances with Missing Values
Removed instances with missing values for status, country_code, category_code, and founded_at to maintain consistency.
    b. Delete Outliers
Identified and removed outliers in funding_total_usd and funding_rounds using the Interquartile Range (IQR) method.
    c. Delete Contradictory Data
Removed data points that showed contradictory information or inconsistencies.

After preprocessing, the DataFrame has the following information:

Total columns: 11
Non-Null Count: 63585
Data types: float64(7), object(4)
![image](https://github.com/akole-Pratik/Startup-Acquisition-Status-Modeling-Using-Machine-Learning-Internship-Project-/assets/81501071/e0b1fb71-3fb6-42db-8029-5e265f4ff56c)

##Exploratory Data Analysis (EDA)
###Univariate & Bivariate Analysis
The Univaraite & Bivariate Analysis phases involved exploring relationships between variables in the dataset. Key visualizations and analyses conducted during this phase include:

Visualization of the distribution of the Status column, which is the target variable, using a horizontal bar plot.
Visualization of the distribution of Milestones using a histogram.
Exploring the relationship between Status and Milestones using a violin plot.
Visualization of the average funding amount by Status using a bar chart.
Exploring the relationship between Status and Funding Total (USD) using a violin plot.
These visualizations provide insights into how different variables interact with each other and their potential impact on the target variable.

##Feature

