from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from sklearn import datasets
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse

from shutil import copyfileobj
from six.moves import urllib
from sklearn.datasets.base import get_data_home
import os
from scipy.io import loadmat

def fetch_mnist(data_home=None):
	mnist_alternative_url = "https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"
	data_home = get_data_home(data_home=data_home)
	data_home = os.path.join(data_home, 'mldata')
	if not os.path.exists(data_home):
		os.makedirs(data_home)
	mnist_save_path = os.path.join(data_home, "mnist-original.mat")
	if not os.path.exists(mnist_save_path):
		mnist_url = urllib.request.urlopen(mnist_alternative_url)
		with open(mnist_save_path, "wb") as matlab_file:
			copyfileobj(mnist_url, matlab_file)

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to the output loss/accuracy plot")

args = vars(ap.parse_args())

print("[INFO] loading MNIST (full) dataset...")
try:
	dataset = datasets.fetch_mldata("MNIST original", data_home="./scikit_learn_data")
except urllib.error.HTTPError as ex:
	fetch_mnist("./scikit_learn_data")
	mnist_raw = loadmat("./scikit_learn_data/mldata/mnist-original.mat")
	mnist = {
		"data": mnist_raw["data"].T,
		"target": mnist_raw["label"][0],
		"COL_NAMES": ["label", "data"],
		"DESCR": "mldata.org dataset: mnist-original"
	}


data = dataset.data.astype("float") / 255.0
(trainX, testX, trainY, testY) = train_test_split(data, dataset.target, test_size=0.25)

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

model = Sequential()
model.add(Dense(256, input_shape=(784,), activation="sigmoid"))
model.add(Dense(128, activation="sigmoid"))
model.add(Dense(10, activation="softmax"))

print("[INFO] training network...")
sgd = SGD(0.01)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=100, batch_size=128)

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=[str(x) for x in lb.classes_]))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])