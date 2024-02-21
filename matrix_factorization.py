#############################
# Model-Based Collaborative Filtering: Matrix Factorization
#############################

# !pip install surprise
import pandas as pd
from surprise import Reader, SVD, Dataset, accuracy
from surprise.model_selection import GridSearchCV, train_test_split, cross_validate
pd.set_option('display.max_columns', None)

# Step 1: Preparing the Data Set
# Step 2: Modeling
# Step 3: Model Tuning
# Step 4: Final Model and Prediction

#############################
# Step 1: Preparing the Data Set
#############################

movie = pd.read_csv(r'dataset\movie_lens_dataset\movie.csv')
rating = pd.read_csv(r'dataset\movie_lens_dataset\rating.csv')
df = movie.merge(rating, how="left", on="movieId")
df.head()

movie_ids = [130219, 356, 4422, 541]
movies = ["The Dark Knight (2011)",
          "Cries and Whispers (Viskningar och rop) (1972)",
          "Forrest Gump (1994)",
          "Blade Runner (1982)"]

sample_df = df[df.movieId.isin(movie_ids)]  
sample_df.head()
sample_df.shape

user_movie_df = sample_df.pivot_table(index=["userId"], columns=["title"], values="rating")

user_movie_df.shape
# Rows represent users, columns represent movies


# It asks for the scale of Surprise rates, to understand how it can be calculated.
# You need to tell the reader that the scale is between 1 and 5
reader = Reader(rating_scale=(1, 5))

# Surprise --> Convert DataFrame to the data structure I specifically use
data = Dataset.load_from_df(sample_df[['userId', 'movieId', 'rating']], reader)

##############################
# Step 2: Modeling
##############################

trainset, testset = train_test_split(data, test_size=.25)
svd_model = SVD()
svd_model.fit(trainset)
predictions = svd_model.test(testset)


accuracy.rmse(predictions)  #0.9362081477697634



svd_model.predict(uid=1.0, iid=541, verbose=True)
svd_model.predict(uid=1.0, iid=356, verbose=True)
sample_df[sample_df["userId"] == 1]
# When any user's ID and movie ID are entered, we will get how many points they can give.

##############################
# Adım 3: Model Tuning
##############################


param_grid = {'n_epochs': [5, 10, 20],
              'lr_all': [0.002, 0.005, 0.007]}

gs = GridSearchCV(SVD,
                  param_grid,
                  measures=['rmse', 'mae'],
                  cv=3,
                  n_jobs=-1,
                  joblib_verbose=True)
gs.fit(data)
gs.best_score['rmse']   # 0.9306106429233026
gs.best_params['rmse']  # bu sonucu veren en iyi parametreler nelerdir

##############################
# Step 4: Final Model and Prediction
##############################
# The default parameters of the model were different, the ones we found are different, the model should be created using the new ones when creating the model.


dir(svd_model)
svd_model.n_epochs

# Rebuilding the model with the best parameters from grid search
svd_model = SVD(**gs.best_params['rmse'])


#bütün veriseti build_full_trainset'ne çevrildi, yeni modeli bütün veriye uygulayalım
data = data.build_full_trainset()
svd_model.fit(data)

svd_model.predict(uid=1.0, iid=541, verbose=True)






