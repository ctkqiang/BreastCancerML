#!/usr/bin/env python
Copyright = """
                  Copyright 2020 © John Melody Me

      Licensed under the Apache License, Version 2.0 (the "License");
      you may not use this file except in compliance with the License.
      You may obtain a copy of the License at

                  http://www.apache.org/licenses/LICENSE-2.0

      Unless required by applicable law or agreed to in writing, software
      distributed under the License is distributed on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
      See the License for the specific language governing permissions and
      limitations under the License.
      @Author : John Melody Me
      @Copyright: John Melody Me & Tan Sin Dee © Copyright 2020
      @INPIREDBYGF: Cindy Tan Sin Dee <3
"""

import numpy as np
import pandas as pd
from pandas import *
import matplotlib.pyplot as plt
from matplotlib import cm as cm
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

print(Copyright)
breastCancerData = pd.read_csv("data/breastCancerData.csv", index_col=False)
breastCancerData.head(5)
# print(breastCancerData.head(5), "\n", breastCancerData.shape)
print(breastCancerData.describe())
breastCancerData["diagnosis"] = breastCancerData["diagnosis"].apply(lambda x : "1" if x == "M" else "0")
breastCancerData = breastCancerData.set_index("id")
del breastCancerData["Unnamed: 32"]
# print(breastCancerData.groupby("diagnosis").size())
"""
OUTPUT:

                 id  radius_mean  texture_mean  ...  symmetry_worst  fractal_dimension_worst  Unnamed: 32
count  5.690000e+02   569.000000    569.000000  ...      569.000000               569.000000          0.0
mean   3.037183e+07    14.127292     19.289649  ...        0.290076                 0.083946          NaN
std    1.250206e+08     3.524049      4.301036  ...        0.061867                 0.018061          NaN
min    8.670000e+03     6.981000      9.710000  ...        0.156500                 0.055040          NaN
25%    8.692180e+05    11.700000     16.170000  ...        0.250400                 0.071460          NaN
50%    9.060240e+05    13.370000     18.840000  ...        0.282200                 0.080040          NaN
75%    8.813129e+06    15.780000     21.800000  ...        0.317900                 0.092080          NaN
max    9.113205e+08    28.110000     39.280000  ...        0.663800                 0.207500          NaN

[8 rows x 32 columns]
diagnosis
0    357
1    212
dtype: int64

"""
plotTitle = "Breast Cancer Attributes Correlation"
breastCancerData.plot(kind="density", subplots=True, layout=(5, 7),
sharex=False, legend= False, fontsize=1)
# plt.show()
fig = plt.figure()
ax1 = fig.add_subplot(111)
cmap = cm.get_cmap("jet", 30)
cax = ax1.imshow(breastCancerData.corr(), interpolation="none", cmap=cmap)
ax1.grid(True)
plt.title(plotTitle)
fig.colorbar(cax, ticks=[.75,.8,.85,.90,.95,1])
# plt.show()

x = breastCancerData.drop("diagnosis", axis=1).values
y = breastCancerData["diagnosis"].values
# print("x: ", x, "\n\n" "y: ", y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.20, random_state=21)

# Baseline Algorithm Inspection:
model_list = []
model_list.append(("CART", DecisionTreeClassifier()))
model_list.append(("SVM", SVC()))
model_list.append(("NB", GaussianNB()))
model_list.append(("KNN", KNeighborsClassifier()))

num_folds = 10
RESULTS = []
names = []
randomState = 123

for name, model in model_list:
      kfold = KFold(n_splits=num_folds, random_state=randomState)
      start = time.time()
      CV_RESULTS = cross_val_score(model, x_train, y_train, cv=kfold, scoring="accuracy")
      end = time.time()
      RESULTS.append(CV_RESULTS)
      names.append(name)
      print( "%s: %f (%f) (run time: %f)" % (name, CV_RESULTS.mean(), CV_RESULTS.std(), end-start))

fig = plt.figure()
fig.suptitle('Performance Comparison')
ax = fig.add_subplot(111)
plt.boxplot(RESULTS)
print(RESULTS)
ax.set_xticklabels(names)
plt.show()