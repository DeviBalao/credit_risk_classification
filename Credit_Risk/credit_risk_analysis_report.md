# Credit Risk Analysis Report

## Overview of the Analysis

The purpose of the analysis is to build, train and evaluate a model based on loan risk and to identify the creditworthiness of borrowers. The following features are used to predict the loan status i.e. whether it is a healthy or high-risk loan:

  * loan size
  * interest rate
  * borrower income
  * debt to income
  * number of accounts
  * derogatory marks 
  * total debt
  
### Steps in the machine learning process for analysis:

1. Load the input data (from csv file) to a DataFrame.
2. From the dataframe, identify the features (X) and target column (y).
3. Check the balance of the two possible outcomes in the target column (y) using value_counts(). 
4. Split the dataset into training and testing datasets using train_test_split()
5. Standardize the feature values using StandarScaler().fit on the training data and then transform the test and training data using transform().
	  Standardization is done so all the values will be in the same range and help in uniform weight for each feature.
6. Create a LogisticRegression model with random_state = 1. 
	  random_state is used to get consistent result each time this model is run.
7. Fit this model on training data (Features and target column)
8. Use this model and make prediction for test data.
9. Compare the values of the target column in the test data, with the predicted values.
10. Evaluate the model by analyzing various metrics like the balanced accuracy score, confusion matrix and classification reports.

## Results

* Machine Learning Model 1: LogisticRegression
  - This model has an accuracy of 99% in predicting healthy and high-risk loans. 
  - Precision and recall for predicting healthy loans is 100%. 
  - For predicting high-risk loans, precision is 87% and recall is 98%. 


* Machine Learning Model 2: RandomOverSampler - 
  This is a Logistic Regression model with resampled data. The sample data has equal number of records for both possible options of the target column.
    - This model has an accuracy of 99% in predicting healthy and high-risk loans.  
    - Precision of 100% and recall of 99% for healthy loans.
    - Precision of 87% and recall of 100% for high-risk loans.

## Summary
Both the models have same accuracy and precision. RandomOverSampler may have overfiitted the data, so to choose one of these two models, will go with LogisticRegression model (Model 1).

