#############################
# Model-Based Collaborative Filtering: Matrix Factorization
#############################

# !pip install surprise
import pandas as pd
from surprise import Reader, SVD, Dataset, accuracy
from surprise.model_selection import GridSearchCV, train_test_split, cross_validate
pd.set_option('display.max_columns', None)

# Adım 1: Veri Setinin Hazırlanması
# Adım 2: Modelleme
# Adım 3: Model Tuning
# Adım 4: Final Model ve Tahmin

#############################
# Adım 1: Veri Setinin Hazırlanması
#############################

movie = pd.read_csv(r'C:\Users\MerveATASOY\Desktop\data_scientist_miuul\egitim_teorik_icerikler\Bolum_6_Tavsiye_Sistemleri\dataset\movie_lens_dataset\movie.csv')
rating = pd.read_csv(r'C:\Users\MerveATASOY\Desktop\data_scientist_miuul\egitim_teorik_icerikler\Bolum_6_Tavsiye_Sistemleri\dataset\movie_lens_dataset\rating.csv')
df = movie.merge(rating, how="left", on="movieId")
df.head()

movie_ids = [130219, 356, 4422, 541]
movies = ["The Dark Knight (2011)",
          "Cries and Whispers (Viskningar och rop) (1972)",
          "Forrest Gump (1994)",
          "Blade Runner (1982)"]

sample_df = df[df.movieId.isin(movie_ids)]   #df içerisindeki movieid'lerde şunlar şunlar var mı, varsa bunları seç
sample_df.head()

sample_df.shape

user_movie_df = sample_df.pivot_table(index=["userId"],
                                      columns=["title"],
                                      values="rating")

user_movie_df.shape
#satırlar kullanıcıları sutunlar filmleri ifade ediyor

#surprise rate'lerin scalasını ister, neye göre hesap yapabileceğini anlamak için
#reader'a scalanın 1 ile 5 arasında olduğunu vermen lazım
reader = Reader(rating_scale=(1, 5))

# surprise : df'i benim özel olarak kullandığım veri yapısına dönüştür ver
data = Dataset.load_from_df(sample_df[['userId',
                                       'movieId',
                                       'rating']], reader)

##############################
# Adım 2: Modelleme
##############################

trainset, testset = train_test_split(data, test_size=.25)
svd_model = SVD()
svd_model.fit(trainset)
predictions = svd_model.test(testset)
# ağırlıklar kullanılarak test setinin değerleri tahmin edilmeye çalışılacak
# pred'de userid ve movie'lerin gerçek ve tahmin edilen değerleri döndü
# r_ui : kullanıcınn verdiği gerçek puan, modelin tahmini : 4.07
# kullanıcı gerçek değerlerine ve tahmin değerlerine yönelik bazı hatalar var


accuracy.rmse(predictions)  #0.9362081477697634
#bir filme verilen puanı tahmin etmek istediğimde yapmam beklenilen ortamala hatadır bu değer



svd_model.predict(uid=1.0, iid=541, verbose=True)

svd_model.predict(uid=1.0, iid=356, verbose=True)


sample_df[sample_df["userId"] == 1]
#bu şekile istediğimiz herhangibir kulanıcnın id'si ve film id'si girildiğinde kaç puan verebileceklerini elde etmiş olacağız


##############################
# Adım 3: Model Tuning
##############################
# kullanılan modelin optimize edilmesi : modelin tahmin performansını artırmaya çalışmak
# hyper parameters optimize edilecek
# 1) epoch sayısı
# 2) learning rate
# tune edilen parametreler sonucunda rmse değerine bakılır eğer düşüyorsa devam edilir
# epoch'ları 5, 10, 20 yap, lr : 0.002, 0.005, 0.007 yap ve olası kombinasyonları dene

param_grid = {'n_epochs': [5, 10, 20],
              'lr_all': [0.002, 0.005, 0.007]}

# mae : gerçek değerler - tahmin edilen değerler karelerinin ortalaması
# gerçek değerler - tahmin edilen değerler karelerinin ortalamanın karekökünü al
# 3 katlı çapraz doğrulama yap  cv= 3 , veri setini 3e böl, 2 parçası ile model kur 1 parçası ile test et
# sonra diğer parçası diğer parçası ile model kur diğer 1 parça ile test et, sonra kalan diğer 2 parçası ile model kur ve dışarıda kalan ile test et
# ve bu test işlemlerinin ortalaması al
# n_jobs=-1 işlemcileri tüm performans ile kullan
# joblib_verbose=True işlemler gerçekleşirken raporlama yap

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
# Adım 4: Final Model ve Tahmin
##############################
#modelin ön tanımlı paraemtreleri farklıydı bizim bulduklarımız farklı, model oluştururken bu parametreler kullanılarak model oluşturulmalı


dir(svd_model)
svd_model.n_epochs

#grid search'ten gelen en iyi parametrelerle yeniden model kuruluyor
svd_model = SVD(**gs.best_params['rmse'])
#gelen parametre değerleri SVD'ye el ile girilebilir, bunu daha hızlı girmenin bir yolu ise keyworded arguman kullanmaktır ** ile

#bütün veriseti build_full_trainset'ne çevrildi, yeni modeli bütün veriye uygulayalım
data = data.build_full_trainset()
svd_model.fit(data)

svd_model.predict(uid=1.0, iid=541, verbose=True)






