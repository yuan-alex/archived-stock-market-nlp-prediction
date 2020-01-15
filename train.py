import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from termcolor import colored
from pymagnitude import *
import datetime
import string
from numba import jit

from joblib import dump, load

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

filename = input("Please enter numpy array name: ")

try:
	X_train = np.load(filename)
	print(colored("[FILE] File has been successfully opened", "green"))
	print(X_train[0].shape)
except Exception as e:
	print(colored("[FILE] Error, couldn't open your file, error: " + e, "red"))
	quit()

MAX_INDEX = 1989

# again, feel free to use your own data
# not sure how it will affect the rest of the program though
df = pd.read_csv("Data/Combined_News_DJIA.csv")
df_DJIA = pd.read_csv("Data/DJIA_table.csv")

df_y = df["Label"]
y_train_np = df_y.values[:MAX_INDEX]

y_train = y_train_np
y_train = y_train_np[:, None]

word_vec = Magnitude("word2vec/GoogleNews-vectors-negative300.magnitude")

def build_model():
	model = keras.Sequential()

	model.add(layers.Dense(8, activation="relu", input_shape=(7500,)))
	model.add(layers.Dense(8, activation="relu"))
	model.add(layers.Dropout(0.5))
	model.add(layers.Dense(8, activation="relu"))
	model.add(layers.Dense(8, activation="relu"))
	model.add(layers.Dense(1, activation="sigmoid"))

	model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(lr=0.0001), metrics=["accuracy"])
	return model

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_train[:], y_train[:], test_size=0.2, random_state=69)
train = "nn"
## nn for neural network with keras, sk for scikit-learn algorithms

if train == "nn":
	model = build_model()
	print(model.summary())
	history = model.fit(X_train, y_train, batch_size=4, epochs=300, validation_data=(X_test, y_test))
	model.evaluate(X_test, y_test)
	model.save("model.h5")

	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('Model accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.show()

	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.show()

	for i in range(0, 50):
		print(colored(round(model.predict(np.expand_dims(X_train[i], axis=0))[0][0]), "green"), end="    ")
	print("")


elif train == "sk":
	# select whichever one you would like to use

	from sklearn.linear_model import LogisticRegression
	from sklearn.svm import LinearSVC
	from sklearn.ensemble import GradientBoostingClassifier
	from sklearn.gaussian_process import GaussianProcessClassifier
	from sklearn.ensemble import AdaBoostClassifier

	print(colored("[TRAIN] Training with sklearn", "green"))
	model = GaussianProcessClassifier()
	model.fit(X_train, y_train)
	print(colored("[TRAIN] sklearn complete", "green"))
	score = model.score(X_test, y_test)
	print(colored("Accuracy: {}".format(score), "cyan",attrs=['bold']))
	dump(model, 'model.joblib')

	for i in range(0, 50):
		print(colored(model.predict(np.expand_dims(X_train[i], axis=0)), "green"), end="    ")
	print("")
