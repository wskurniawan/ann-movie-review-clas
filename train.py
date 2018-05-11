import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns

#memuat data
def load_directory_data(directory):
   # deklarasi objek json dengan atribut sentence dan sentiment berupa array
   data = {}
   data["sentence"] = []
   data["sentiment"] = []
   for file_path in os.listdir(directory):
      with tf.gfile.GFile(os.path.join(directory, file_path), "r") as f:
         data["sentence"].append(f.read())
         data["sentiment"].append(
             re.match("\d+_(\d+)\.txt", file_path).group(1))
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
   train_df = load_dataset(os.path.join(os.path.dirname("dataset"), "train"))
   test_df = load_dataset(os.path.join(os.path.dirname("dataset"), "test"))

   return train_df, test_df


train_df, test_df = set_dataset()
train_df.head()

# Training input on the whole training set with no limit on training epochs.
train_input_fn = tf.estimator.inputs.pandas_input_fn(
    train_df, train_df["polarity"], num_epochs=None, shuffle=True)

# Prediction on the whole training set.
predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(
    train_df, train_df["polarity"], shuffle=False)
# Prediction on the test set.
predict_test_input_fn = tf.estimator.inputs.pandas_input_fn(
    test_df, test_df["polarity"], shuffle=False)

embedded_text_feature_column = hub.text_embedding_column(
    key="sentence", 
    module_spec="https://tfhub.dev/google/nnlm-en-dim128/1")

estimator = tf.estimator.DNNClassifier(
    hidden_units=[500, 100],
    feature_columns=[embedded_text_feature_column],
    n_classes=2,
    optimizer=tf.train.AdagradOptimizer(learning_rate=0.003))

# Training for 1,000 steps means 128,000 training examples with the default
# batch size. This is roughly equivalent to 5 epochs since the training dataset
# contains 25,000 examples.
estimator.train(input_fn=train_input_fn, steps=1000)

train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
test_eval_result = estimator.evaluate(input_fn=predict_test_input_fn)

print("Training set accuracy: {accuracy}".format(**train_eval_result))
print("Test set accuracy: {accuracy}".format(**test_eval_result))

train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
test_eval_result = estimator.evaluate(input_fn=predict_test_input_fn)

print("Training set accuracy: {accuracy}".format(**train_eval_result))
print("Test set accuracy: {accuracy}".format(**test_eval_result))