import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from termcolor import colored
from pymagnitude import *
import datetime
import string

from joblib import dump, load

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

word_vec = Magnitude("word2vec/GoogleNews-vectors-negative300.magnitude")
model = keras.models.load_model("model.h5")

def filter_byte(word):
	if (word[:2] == "b'" or word[:2] == 'b"') and (word[-1] == "'" or word[-1] == '"'):
		return word[2:-1]
	else:
		return word

filters = '!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
def remove_punctuation(word):
	return word.translate(str.maketrans('', '', filters)).lower()

def average_query(query):
	return np.mean(query, axis=0)

common = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']

def remove_common(query):
	return [i for i in query if not(i in common)]


print(colored("MIT License, NO LIABILITY.", "red"))
print(colored("\nPlease enter the top 25 news titles from your source below (Reddit worldnews preferred):\n", "white"))

predict_X = np.zeros(shape=(1, 7500))

t = np.array([])
for j in range(0, 25):
	user_input = input(colored("Please enter top article #{}: ".format(j + 1), "blue", attrs=["bold"]))
	t = np.append(t, np.mean(word_vec.query(remove_common(remove_punctuation(filter_byte(str(user_input))).split(" "))), axis=0))
predict_X[0] = t

prediction = model.predict(predict_X)[0][0]

print("")
print(colored("There is a {}% chance the Dow Jones Index will increase today.".format(prediction * 100), "green", attrs=["bold"]))
print("")
