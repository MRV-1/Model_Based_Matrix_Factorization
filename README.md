# # Model-Based Collaborative Filtering: Matrix Factorization
A proposal has been developed based on the training and prediction processes of SVD, one of the machine learning models 🤖

### Step 1: Preparing the Data Set
### Step 2: Modeling
### Step 3: Model Tuning
### Step 4: Final Model and Prediction



## Step 1: Preparing the Data Set 👩‍🌾
*** Surprise library must be downloaded for model selection
!pip install surprise

--> For dataset : https://grouplens.org/datasets/movielens/

# Step 2: Modeling 💥

--> This is the stage where the values of the test set are estimated using weights.

--> Actual and predicted values of userid and movies are returned in Predictions
--> r_ui parameter is the actual score given by the user

--> Some errors were detected regarding user actual values and predicted values

# Adım 3: Model Tuning 💣

--> Optimizing the model used: Increases the prediction performance of the model
--> Hyper parameters optimized 🤖

1) number of epochs 
2) learning rate

--> As a result of the tuned parameters, the RMSE value is checked and if it decreases, continue.

--> make epochs 5, 10, 20, lr : 0.002, 0.005, 0.007 and try possible combinations


**********
 Mean Absolute Error : actual values - average of squares of predicted values
 actual values - take the square root of the mean of the squares of the predicted values
 
Perform 3-fold cross validation cross validation= 3 ,

k-fold Cross Validation: Divide the data set into 3, build a model with 2 parts, test it with 1 part, then build a model with the other part, test it with the other part, then build a model with the remaining 2 parts and test it with the remaining part

And take the average of these test runs

n_jobs=-1 use processors with full performance

joblib_verbose=True report while transactions are taking place

# Step 4: Final Model and Prediction 🎯

The default parameters of the model were different, the ones we found are different, the model should be created using the new ones when creating the model.

Rebuilt model with best parameters from grid search.











