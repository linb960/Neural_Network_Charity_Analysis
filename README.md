# Neural Network Charity Analysis
## Overview 
This analysis will use neural networks with a binary classifier that will attempt to predict whether applicants will be successful if funded by a charitable organization, Alphabet Soup.

## RESULTS 
### PREPROCESSING
The data that was provided for this analysis is a comma separated value, csv, file.  Within the file the columns are:
- EIN and NAME—Identification columns
- APPLICATION_TYPE—Alphabet Soup application type
- AFFILIATION—Affiliated sector of industry
- CLASSIFICATION—Government organization classification
- USE_CASE—Use case for funding
- ORGANIZATION—Organization type
- STATUS—Active status
- INCOME_AMT—Income classification
- SPECIAL_CONSIDERATIONS—Special consideration for application
- ASK_AMT—Funding amount requested
- IS_SUCCESSFUL—Was the money used effectively

#### TARGET
The target is the __IS_SUCCESSFUL__ column.  This is in line with trying to predict if an applicant is successful with money funded.

#### FEATURES
The columns used for the features are __APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS and ASK_AMT.

#### DROPPED 
__EIN and Name__ are dropped as they don't add anything to the analysis.

#### BINNING
__APPLICATION_TYPE and CLASSIFICATION__ have more than 10 unique values so a density plot was created to determine the distribution of the column values. The APPLICATION_TYPE's with fewer than 500 were binned together and the CLASSIFICATION's below 1000 were binned together.

#### ENCODING
The categorical variables __APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, INCOME_AMT and SPECIAL_CONSIDERATIONS__ are encoded using one-hot encoding.  They are added to a new DataFrame and then this DataFrame and original DataFrame are merged and the original columns are dropped.

#### SPLIT DATA
The data is now ready for splitting:
```
# Split our preprocessed data into our features and target arrays
y = application_df.IS_SUCCESSFUL.values
X = application_df.drop("IS_SUCCESSFUL", axis=1).values

# Split the preprocessed data into a training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y)
```
#### AND SCALING
```
from sklearn.preprocessing import StandardScaler
# Create a StandardScaler instances
scaler = StandardScaler()

# Fit the StandardScaler
X_scaler = scaler.fit(X_train)

# Scale the data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
```

Compiling, Training, and Evaluating the Model
How many neurons, layers, and activation functions did you select for your neural network model, and why?
Were you able to achieve the target model performance?
What steps did you take to try and increase model performance?
Summary: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and explain your recommendation.
