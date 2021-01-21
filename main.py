# Soner Karaevli - 30716005

# csv dosyalarını okumak için gereken kütüphane
import pandas as pd

# Dataseti içe aktarıyoruz, text ve label'ı  değişkene atıyorum.
news = pd.read_csv('data.csv')
X = news['text']
y = news['label']


#data seti bölmek için kullanılan kütüphane
from sklearn.model_selection import train_test_split
# Dataseti eğitmek için test ve train şeklinde bölüyorum
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22)



#pipeline oluşturmak için kullanılan kütüphane
from sklearn.pipeline import Pipeline

#verileri serileştirmek için kullanılan kütüphaneyi ekliyorum
from sklearn.feature_extraction.text import TfidfVectorizer

# Naive Bayes Kütüphanelerini ekliyorum
from sklearn.naive_bayes import MultinomialNB, GaussianNB , BernoulliNB

# TfidfVectorizer kütüphanesi data'yı seri hale getiriyor.
# Pipeline oluşturuyorum ve içinde gereksiz kelimeleri stopwords ile çıkarıyorum. Daha sonra Naive Bayes uyguluyorum.
seri = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),
                 ('bernoli', BernoulliNB()),
                 # ('nbmodel1',  MultinomialNB()),
                 ])

# Makineyi Eğitiyorum
seri.fit(X_train, y_train)

# Test verileri için tahmin yapiyorum
tahmin = seri.predict(X_test)


# basari oranı için gereken kütüphaneyi ekliyorum.
from sklearn.metrics import accuracy_score
# Başarı Oranı Hesaplıyorum
basari = accuracy_score(y_test, tahmin)


#performansı kontrol etmek için kullanılan kütüphaneyi ekliyorum.
from sklearn.metrics import classification_report, confusion_matrix
# Modelimizin performansını kontrol ediyorum.
print(classification_report(y_test, tahmin))


# karmaşıklık matrisi hesapliyorum
print("Karmasiklik Matrisi: ")
print(" ")
print(confusion_matrix(y_test,tahmin))
print(" ")
print("Basari Orani: ")
print(basari)