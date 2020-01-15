import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from termcolor import colored
from pymagnitude import *
import datetime
import string
from tqdm import tqdm

from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence

# different csv files have different formats of data
# feel free to change this to one which fits your file
MAX_INDEX = 1989

# feel free to use your own data by changing this
# you might have to change some parts of the program however
try:
	df = pd.read_csv(input("Please enter the directory in which the CSV file is located: "))
except Exception as error:
	print(colored("[DATA] Could not save array error: " + error, "red"))

df_y = df["Label"]

# different csv files have different formats of data
# feel free to change this to one which fits your file
df_X = df.iloc[:, 2:27]

y_train_np = df_y.values[:MAX_INDEX]

y_train = y_train_np
y_train = y_train_np[:, None]

word_vec = Magnitude("word2vec/GoogleNews-vectors-negative300.magnitude")

def filter_byte(word):
	if (word[:2] == "b'" or word[:2] == 'b"') and (word[-1] == "'" or word[-1] == '"'):
		return word[2:-1]
	else:
		return word

common = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']

def remove_common(query):
	return [i for i in query if not(i in common)]

import nltk
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
def remove_stop(query):
	return [w for w in query if not(w in stop_words)]

X_train_np = df_X.copy().values[:MAX_INDEX]
X_train = np.zeros(shape=(MAX_INDEX, 7500))

print(colored("[STARTING] Processing started at {}\n".format(datetime.datetime.now()), "green"))
for i in tqdm(range(0, MAX_INDEX)):
	row_vector = np.array([])
	for j in range(0, 25):
		row_vector = np.append(row_vector, np.mean(word_vec.query(remove_stop(remove_common(text_to_word_sequence(filter_byte(str(X_train_np[i][j])))))), axis=0))
	X_train[i] = row_vector

print(colored("[COMPLETE] Processing has been finished", "green"))

filename = input("Please enter a filename (blank for training_data.npy): ")
if filename == "":
	filename = "training_data.npy"

try:
	np.save(filename, X_train)
	print(colored("[SAVE] Save complete", "green"))
except Exception as error:
	print(colored("[SAVE] Could not save array error: " + error + "\nAttempting to save backup...", "red"))
	np.save("BACKUP_SAVE.npy", X_train)
