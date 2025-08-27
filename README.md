# **Project Title: Customer Attrition Analysis**

## **Overview:**
This project uses bank churners dataset that is used to predict the Attrition Flag status of customers based on some features.
This is a classification analysis

## **Objectives:**
* *Visualize data*
* *Make Inferences*
* *Technique to handle imbalanced dataset*


## Information
**Source:** [Bank Customer Attrition Dataset](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers)  
**Shape:** 10,000+ rows × 20+ columns  
**Key Features:**
- `Customer_Age`, `Gender`, `Marital_Status`
- `Education_Level`, `Income_Category`
- `Credit_Limit`, `Avg_Utilization_Ratio`
- `Attrition_Flag` (Target)

## Note:
dropped the last two columns named:

'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1'

and

'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'

These were not part of original dataset 

## Preprocessing and EDA

* *Different plots to visualize data and relationships between them*
* *OneHot Encoding on **Education_Level**, **Marital_Status**, **Card_Category***
* *Label Encoding on **Gender**, **Attrition_Flag***
* *Ordinal on **Income_Category***

## Technology used:
* Python(pandas, matplotlib, seaborn, scikit-learn, imbalanced-learn)
* VsCode
* Git/GitHub



