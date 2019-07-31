import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from BinaryClassifier import BinaryClassifier

train = pd.read_csv("./Dataset/train.csv").drop(["PassengerId", "Name", "Ticket", "Embarked", "Cabin"], axis=1)
test = pd.read_csv("./Dataset/test.csv").drop(["PassengerId", "Name", "Ticket", "Embarked", "Cabin"], axis=1)

train_labels = train["Survived"]

def gender_to_int(sample): return int(sample == "male")
train_gender = [[gender_to_int(sample)] for sample in train["Sex"].values]
test_gender = [[gender_to_int(sample)] for sample in test["Sex"].values]

train_features = np.concatenate([train.drop(["Sex"], axis=1).values, train_gender], axis=1)
test_features = np.concatenate([test.drop(["Sex"], axis=1).values, test_gender], axis=1)

train_ds = tf.data.Dataset.from_tensor_slices((train_features, train_labels)).shuffle(100)
val_ds = train_ds.take(200).batch(200)
train_ds = train_ds.skip(200).batch(64)
test_ds = tf.data.Dataset.from_tensor_slices((test_features)).batch(64)

model = BinaryClassifier([8, 4, 1])
model.compile(tf.keras.optimizers.Adadelta(0.0001), tf.keras.losses.MeanSquaredError(), [tf.keras.metrics.Accuracy()])
model.fit(train_ds.take(1), epochs=20, verbose=2, validation_data=val_ds)

#for sample in train_ds.take(1):
#    print(sample)

