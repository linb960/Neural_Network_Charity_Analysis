# Neural Network Charity Analysis
## OVERVIEW
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

### Compiling, Training, and Evaluating the Model
Results were exported to two HDF5 files.  AlphabetSoupCharity.h5 and AlphabetSoupCharity_Optimazation.h5
For Deliverable 2 and 3 Each attempt created checkpoint files and can be found https://github.com/linb960/Neural_Network_Charity_Analysis/tree/main/checkpoints

#### Deliverable 2 Attempt https://github.com/linb960/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity_Optimazation.ipynb
With 8 nodes in the first hidden layer and 5 nodes in the second hidden layer.  The Activation used is Relu in the hidden layers and Sigmoid in the output. 
```
# Define the model - deep neural net
number_input_features = len(X_train[0])
hidden_nodes_layer1 =  8
hidden_nodes_layer2 = 5

nn = tf.keras.models.Sequential()

# First hidden layer
nn.add(
    tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation="relu")
)

# Second hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation="relu"))


# Output layer
nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

# Check the structure of the model
nn.summary()
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 8)                 352       
_________________________________________________________________
dense_1 (Dense)              (None, 5)                 45        
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 6         
=================================================================
Total params: 403
Trainable params: 403
Non-trainable params: 0
```
__Evaluating the model shows a 72.73% accuracy:__
```
model_loss, model_accuracy = nn.evaluate(X_test_scaled,y_test,verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

8575/1 - 0s - loss: 0.5303 - accuracy: 0.7273
Loss: 0.5555491790618563, Accuracy: 0.7273469567298889
```


#### Second Attempt https://github.com/linb960/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity_Optimazation.ipynb. Line 20

Nodes are changed to 16 in first hidden layer, 8 nodes in the second hidden layer and a third layer with 6 nodes is also added.  The Activation used is still Relu in the hidden layers and Sigmoid in the output.
```
# Define the model - deep neural net
number_input_features = len(X_train[0])
hidden_nodes_layer1 =  16
hidden_nodes_layer2 = 8
hidden_nodes_layer3 = 6

nn = tf.keras.models.Sequential()

# First hidden layer
nn.add(
    tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation="relu")
)

# Second hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation="relu"))

# Third hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer3, activation="relu"))

# Output layer
nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

# Check the structure of the model
nn.summary()


Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_3 (Dense)              (None, 16)                704       
_________________________________________________________________
dense_4 (Dense)              (None, 8)                 136       
_________________________________________________________________
dense_5 (Dense)              (None, 6)                 54        
_________________________________________________________________
dense_6 (Dense)              (None, 1)                 7         
=================================================================
Total params: 901
Trainable params: 901
Non-trainable params: 0
```
__Evaluating the model shows a 73.06% accuracy:__
```
model_loss, model_accuracy = nn.evaluate(X_test_scaled,y_test,verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

8575/1 - 0s - loss: 0.5063 - accuracy: 0.7306
Loss: 0.5545681785529278, Accuracy: 0.7306122183799744
```
#### Second Attempt https://github.com/linb960/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity_Optimazation_Attempt3.ipynb

The number of Nodes and layers stay the same from the second attempt with 16 in first hidden layer, 8 nodes in the second hidden layer and a third layer with 6 nodes.  The Activation used changes to Tanh in the hidden layers and stays Sigmoid in the output.

__Additional change__ comes by changing the number of features.  By Binning the APPLICATION_TYPE's if there are less than 1000 instead of less than 500 we reduced number of columns by 4 as seen here:
```
T3       27037
Other     2266
T4        1542
T6        1216
T5        1173
T19       1065
Name: APPLICATION_TYPE, dtype: int64
```
Once this was done the model was defined:
```
# Define the model - deep neural net
number_input_features = len(X_train[0])
hidden_nodes_layer1 =  16
hidden_nodes_layer2 = 8
hidden_nodes_layer3 = 6

nn = tf.keras.models.Sequential()

# First hidden layer
nn.add(
    tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation="tanh")
)

# Second hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation="tanh"))

# Third hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer3, activation="tanh"))

# Output layer
nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

# Check the structure of the model
nn.summary()


Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_3 (Dense)              (None, 16)                704       
_________________________________________________________________
dense_4 (Dense)              (None, 8)                 136       
_________________________________________________________________
dense_5 (Dense)              (None, 6)                 54        
_________________________________________________________________
dense_6 (Dense)              (None, 1)                 7         
=================================================================
Total params: 901
Trainable params: 901
Non-trainable params: 0
```
__Evaluating the model shows a 73.03% accuracy:__
```
model_loss, model_accuracy = nn.evaluate(X_test_scaled,y_test,verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

8575/1 - 0s - loss: 0.5037 - accuracy: 0.7303
Loss: 0.5566521305697305, Accuracy: 0.7302623987197876
```

### Summary
