#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 21:59:16 2023

@author: josephinemiller
"""

import warnings
warnings.filterwarnings("ignore")
#  Algorithms
from pandas import read_csv
from matplotlib import pyplot
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np

filePath = 'C:/Users/josephinemiller/Desktop/'
filename = 'wine.csv'
names = ['class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', "Hue", "OD280/OD315 of diluted wines", 'Proline']
df = read_csv('/Users/josephinemiller/Desktop/wine.csv', names=names)

X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values
print("X shape:", X.shape)
print("y shape:", y.shape)

scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)

models = []
models.append(('Linear SVC', OneVsRestClassifier(LinearSVC(C=100, loss='hinge', random_state=1, max_iter=1000000))))
models.append(('Kernel SVC (RBF Kernel, C=1)', OneVsRestClassifier(SVC(kernel='rbf', C=1, random_state=1, max_iter=1000000))))
models.append(('Kernel SVC (RBF Kernel, C=10)', OneVsRestClassifier(SVC(kernel='rbf', C=10, random_state=1, max_iter=1000000))))
models.append(('Kernel SVC (RBF Kernel, C=100)', OneVsRestClassifier(SVC(kernel='rbf', C=100, random_state=1, max_iter=1000000))))
models.append(('Random Forest', RandomForestClassifier(max_depth=2, random_state=1)))
models.append(('Adaboost', AdaBoostClassifier(random_state=1)))

# Plotting the multi-class ROC curve
fig, ax = pyplot.subplots(figsize=(8, 6))

# One-hot encoding of the target variable
y_bin = label_binarize(y, classes=np.unique(y))

for name, model in models:
    classifier = OneVsRestClassifier(model)
    y_score = cross_val_predict(classifier, rescaledX, y_bin, cv=KFold(n_splits=10, random_state=7, shuffle=True))
    
    # Compute ROC curve and ROC-AUC score for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(np.unique(y))):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC-AUC score
    fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Plot ROC curve
    ax.plot(fpr["micro"], tpr["micro"], label=f'{name} (AUC = {roc_auc["micro"]:.2f})')
    print(f'{name} ROC AUC: {roc_auc["micro"]:.4f}')

ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2)  # Diagonal line for random classifier
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Multi-Class ROC Curve')
ax.legend(loc="lower right")
pyplot.show()
