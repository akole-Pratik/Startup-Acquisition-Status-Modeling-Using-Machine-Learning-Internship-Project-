# Startup-Acquisition-Status-Modeling-Using-Machine-Learning-Internship-Project-
Startup Acquisition Status Modeling Using Machine Learning 
# Building-Machine-Learning-Pipeline-on-Startups-Acquisition
## Aim: 
Our aim to understand the financial conditions of company fundraising goals.
##Description
The objective of the project is to predict whether a startup which is currently Operating, IPO, Acquired, or closed.This problem will be solved through a Supervised Machine Learning approach by training a model based on the history of startups which were either acquired or closed. The project aims to predict the acquisition status of startups based on various features such as funding rounds, total funding amount, industry category, and geographic location. By leveraging machine learning pipelines, we preprocess the data, select relevant features, and train models to classify startups into different acquisition status categories. The project utilizes Python libraries such as scikit-learn, pandas, matplotlib, seaborn, joblib, and XGBoost for model development and evaluation. The goal is to provide insights into the factors influencing startup acquisition and build a predictive tool that can assist stakeholders in making informed decisions.

## Understanding the Dataset
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


## Data Preprocessing

# Data Cleaning
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

## Exploratory Data Analysis (EDA)
###Univariate & Bivariate Analysis
The Univaraite & Bivariate Analysis phases involved exploring relationships between variables in the dataset. Key visualizations and analyses conducted during this phase include:

Visualization of the distribution of the Status column, which is the target variable, using a horizontal bar plot.
Visualization of the distribution of Milestones using a histogram.
Exploring the relationship between Status and Milestones using a violin plot.
Visualization of the average funding amount by Status using a bar chart.
Exploring the relationship between Status and Funding Total (USD) using a violin plot.
These visualizations provide insights into how different variables interact with each other and their potential impact on the target variable.

## Feature Engineering
1) Feature Selection: We performed feature selection to choose the most relevant features for our analysis.
2) Creation of New Features: We created new features from the existing dataset to enhance predictive power.
3) Normalization and Scaling: We normalized and scaled numerical features to ensure consistency and comparability.
4) Encoding Categorical Variables: We encoded categorical variables to represent them numerically for model training.
5) Feature Engineering Documentation: We documented the entire feature engineering process for transparency and reproducibility.
### Creation of New Features from Dataset
We conducted various operations to create new features:

Converted the 'founded_at' column to datetime format and extracted the year.
Mapped status values to isClosed values and created a new column.
Performed Min-Max scaling on selected numerical features.
Applied one-hot encoding to 'country_code' and 'category_code' columns.
Label encoded the 'status' column for binary classification.
### Feature Selection using Mutual Information (MI)
We computed mutual information between features and the target variable to identify top-ranked features for model training.

After conducting comprehensive feature engineering, our dataset comp_df has undergone significant transformations. Initially containing 11 columns consisting of 3 categorical variables and 8 numerical variables, it has now expanded to encompass 32 columns while maintaining its original 4682 rows. All variables within comp_df have been converted to numerical format, making them suitable for analytical operations. Our data frame is ready to embark on the next phase of model construction with confidence.

## Model Building
Leading up to the Feature Engineering phase, individual interns diligently prepared their datasets to model startup acquisition statuses. After thorough experimentation and evaluation, three standout models emerged for collaborative refinement by the team.
#### For Binary Classification:
We explored Decision Trees.
We delved into the intricacies of Support Vector Machines (SVM).
#### For Multiclass Classification:
We explored the potentials of Gradient Boosting.
We examined the effectiveness of XGBoost.

After thorough analysis and collaborative discussion, we carefully chose one model for binary classification and another for multiclass classification, prioritizing accuracy. Our selections were SVM for binary classification and XGBoost for multiclass classification.

## Model Evaluation
Every model was extensively evaluated, analyzing metrics like accuracy, precision, recall, and F1-score. This evaluation process led to the development of a detailed classification report, which will be used for additional analysis and improvement.

# Machine Learning Pipelines Building
## 1) Binary Classification Model:
We have developed a binary classification model using Random Forest. This model predicts whether a startup will be acquired or not. It analyzes various features of the startup and determines the likelihood of acquisition.
## 2) Multiclass Classification Model:
Similarly, we have constructed a multiclass classification model using an XGBoost classifier. Unlike the binary model, this classifier predicts multiple classes of startup status: Operating, IPO, Acquired, or Closed. It evaluates various factors to categorize startups into these different status categories.
## 3) Combining Pipelines:
Our primary objective is to create three distinct pipelines:

### i) Binary Classification Pipeline:
This pipeline will encapsulate the process of preparing data, training the Random Forest model, and making predictions on whether a startup will be acquired.

### ii) Multiclass Classification Pipeline:
Similarly, this pipeline will handle data preparation, model training using XGBoost, and predicting the status of startups (Operating, IPO, Acquired, or Closed).

### iii) Combined Pipeline:
The challenge lies in integrating these two models into a single pipeline. We must ensure that the output of the binary classifier is appropriately transformed to serve as input for the multiclass classifier. This combined pipeline will enable us to efficiently predict startup statuses.



