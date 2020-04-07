#!/usr/bin/env python
import numpy as np
import pandas as pd
from pandas import *
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import time

breastCancerData = pd.read_csv("data/breastCancerData.csv", index_col=False)
breastCancerData.head(5)
# print(breastCancerData.head(5), "\n", breastCancerData.shape)
print(breastCancerData.describe())
breastCancerData["diagnosis"] = breastCancerData["diagnosis"].apply(lambda x : "1" if x == "M" else "0")
breastCancerData = breastCancerData.set_index("id")
del breastCancerData["Unnamed: 32"]
print(breastCancerData.groupby("diagnosis").size())