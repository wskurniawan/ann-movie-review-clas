import argparse
import pickle
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

parser = argparse.ArgumentParser()
parser.add_argument('--predict', required = True, help = 'Masukkan kalimat')

args = vars(parser.parse_args())

kalimat = []
kalimat.append(args["predict"])
print(kalimat)

#memuat transormasi rfidf
load_transform = pickle.load(open("transform.sav", "rb"))
#print(load_transform)

#memuat model hasil training
load_model = pickle.load(open("model_final.sav", "rb"))
#print(load_model)

#load data train
train_df = pickle.load(open("data.sav", "rb"))

# tfidf = TfidfVectorizer(min_df=5, encoding='utf-8', ngram_range=(1, 1), stop_words="english")
# train_set = tfidf.fit_transform(train_df["sentence"])
# print(train_set.shape)

arg_vector = load_transform.transform(kalimat)
#print(arg_vector.shape)

result = load_model.predict_proba(arg_vector)
print("Probability: ", result)

result = load_model.predict(arg_vector)
str_res = ""
if(result[0] == 0):
   str_res = "negative"

if(result[0] == 1):
   str_res = "positive"


print("Result -> ", result, " ", str_res)

#X_test = pickle.load(open("xtest.sav", "rb"))
#y_test = pickle.load(open("ytest.sav", "rb"))

#print(arg_vector)

#predictions = load_model.predict(X_test)

#print(confusion_matrix(y_test,predictions))
#print(classification_report(y_test,predictions))