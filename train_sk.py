import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
import pickle

#memuat data
def load_directory_data(directory):
   # deklarasi objek json dengan atribut sentence dan sentiment berupa array
   data = {}
   data["sentence"] = []
   data["sentiment"] = []

   counter = 0
   for file_path in os.listdir(directory):
      txt = open(os.path.join(directory, file_path), "r", encoding="utf8")
      data["sentence"].append(txt.read())
      data["sentiment"].append(
         re.match("\d+_(\d+)\.txt", file_path).group(1))
      txt.close()
      print("File index: ", counter)
      counter = counter + 1
         
   return pd.DataFrame.from_dict(data)

#menggabungkan data kalimat positif dan negaif
def load_dataset(directory):
   pos_data = load_directory_data(os.path.join(directory, "pos"))
   neg_data = load_directory_data(os.path.join(directory, "neg"))
   pos_data["polarity"] = 1
   neg_data["polarity"] = 0

   return pd.concat([pos_data, neg_data]).sample(frac=1).reset_index(drop=True)

#melakukan inisiasi data untuk train dan data untuk test
def set_dataset():
   print(os.path.join(os.path.curdir, "train"))
   train_df = load_dataset(os.path.join(os.path.curdir,"dataset", "train"))
   #test_df = load_dataset(os.path.join(os.path.curdir,"dataset", "test"))

   #final_df = train_df + test_df
   return train_df


train_df = set_dataset()
#menyimpan data yang sudah berbentuk array
pickle.dump(train_df, open("data.sav", "wb"))

print("data disimpan")

tfidf = TfidfVectorizer(min_df=5, encoding='utf-8', ngram_range=(1, 1), stop_words="english")

train_set = tfidf.fit_transform(train_df["sentence"])

#menyimpan transformasi tfdif
filename = "transform.sav"
pickle.dump(tfidf, open(filename, "wb"))

#split data
X_train, X_test, y_train, y_test = train_test_split(train_set, train_df['polarity'], random_state = 0)
print("train len ", X_train.shape)
print("test len ", X_test.shape)

#menyimpan data untuk predict
filename = "xtest.sav"
pickle.dump(X_test, open(filename, "wb"))
filename = "ytest.sav"
pickle.dump(y_test, open(filename, "wb"))


print("training start...")
#train
mlp = MLPClassifier(hidden_layer_sizes=(5, 3), max_iter=5, batch_size=1, verbose=True)
mlp.fit(X_train, y_train)

#menyimpan model
filename = "model.sav"
pickle.dump(mlp, open(filename, "wb"))

predictions = mlp.predict(X_test)

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
