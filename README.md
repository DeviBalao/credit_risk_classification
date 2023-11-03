# credit_risk_classification
Train and evaluate a model based on loan risk.

### Steps in the machine learning process for analysis:

1. Load the input data (from csv file) to a DataFrame.
2. From the dataframe, identify the features (X) and target column (y).
3. Check the balance of the two possible outcomes in the target column (y) using value_counts(). 
4. Split the dataset into training and testing datasets using train_test_split()
5. Standardize the feature values using StandarScaler().fit on the training data and then transform the test and training data using transform().
	  Standardization is done so all the values will be in the same range and help in uniform weight for each feature.
6. Create a LogisticRegression model with random_state = 1. 
	  random_state is used to get consistent result each time the model is run.
7. Fit this model on training data (Features and target column)
8. Use this model and make prediction for test data.
9. Compare the values of the target column in the test data, with the predicted values.
10. Evaluate the model by analyzing various metrics like the balanced accuracy score, confusion matrix and classification reports.
