#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 13:43:22 2023

@author: colin
test de la regression logistic de SPSS de diagnostic
"""

import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

#%%
# =============================================================================
# DeLong test
# =============================================================================
# Copyright 2016-2020 Biomedical Imaging Group Rotterdam, Departments of
# Medical Informatics and Radiology, Erasmus MC, Rotterdam, The Netherlands
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Adopted from https://github.com/yandexdataschool/roc_comparison."""

import numpy as np
import scipy.stats


# AUC comparison adapted from
# https://github.com/Netflix/vmaf/
def compute_midrank(x):
    """Computes midranks.

    Args:
       x - a 1D numpy array
    Returns:
       array of midranks

    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2



def compute_midrank_weight(x, sample_weight):
    """Computes midranks.

    Args:
       x - a 1D numpy array
    Returns:
       array of midranks

    """
    J = np.argsort(x)
    Z = x[J]
    cumulative_weight = np.cumsum(sample_weight[J])
    N = len(x)
    T = np.zeros(N, dtype=np.float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = cumulative_weight[i:j].mean()
        i = j
    T2 = np.empty(N, dtype=np.float)
    T2[J] = T
    return T2



def fastDeLong(predictions_sorted_transposed, label_1_count):
    """Fast DeLong test computation.

    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Oerating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }

    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float)
    ty = np.empty([k, n], dtype=np.float)
    tz = np.empty([k, m + n], dtype=np.float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov



def calc_pvalue(aucs, sigma):
    """Computes log(10) of p-values.

    Args:
       aucs: 1D array of AUCs
       sigma: AUC DeLong covariances
    Returns:
       log10(pvalue)

    """
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)



def compute_ground_truth_statistics(ground_truth, sample_weight=None):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    if sample_weight is None:
        ordered_sample_weight = None
    else:
        ordered_sample_weight = sample_weight[order]

    return order, label_1_count, ordered_sample_weight



def delong_roc_variance(ground_truth, predictions):
    """Computes ROC AUC variance for a single set of predictions.

    Args:
       ground_truth: np.array of 0 and 1
       predictions: np.array of floats of the probability of being class 1

    """
    sample_weight = None
    order, label_1_count, ordered_sample_weight = compute_ground_truth_statistics(
        ground_truth, sample_weight)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count, ordered_sample_weight)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov



def delong_roc_test(ground_truth, predictions_one, predictions_two):
    """Computes log(p-value) for hypothesis that two ROC AUCs are different.

    Args:
       ground_truth: np.array of 0 and 1
       predictions_one: predictions of the first model,
          np.array of floats of the probability of being class 1
       predictions_two: predictions of the second model,
          np.array of floats of the probability of being class 1

    """
    order, label_1_count, _ = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    return calc_pvalue(aucs, delongcov)


#%% import data
df = pd.read_excel('/home/colin/Documents/Serena/Dataset_for_probability_model.xlsx')
df.dropna(inplace=True)

#%%
# =============================================================================
# New test: K-fold cross validation
# =============================================================================
#%%
X = df[['% perivenular les', 'Number_CL', 'number_PRL']]
X_cvs = df[['% perivenular les']]
X_cl = df[['Number_CL']]
X_prl = df[['number_PRL']]
y = df['diagnosis']

# model = LogisticRegression(penalty=None, C=float('inf'), class_weight='balanced')
# scores=[]
aucs = []
accuracys = []
sensitivitys = []
specificitys = []
aucs_cvs = []
accuracys_cvs = []
sensitivitys_cvs = []
specificitys_cvs = []
aucs_cl = []
accuracys_cl = []
sensitivitys_cl = []
specificitys_cl = []
aucs_prl = []
accuracys_prl = []
sensitivitys_prl = []
specificitys_prl = []
model_cvs_ps = []
model_cl_ps = []
model_prl_ps = []
cvs_cl_ps = []
cvs_prl_ps = []
cl_prl_ps = []
kFold=KFold(n_splits=10,random_state=42,shuffle=True)
for train_index,test_index in kFold.split(X):
    # print("Train Index: ", train_index, "\n")
    # print("Test Index: ", test_index)
    
    X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
    X_cvs_train, X_cvs_test = X_cvs.iloc[train_index], X_cvs.iloc[test_index]
    X_cl_train, X_cl_test = X_cl.iloc[train_index], X_cl.iloc[test_index]
    X_prl_train, X_prl_test = X_prl.iloc[train_index], X_prl.iloc[test_index]
    
    model = LogisticRegression(penalty=None, C=float('inf'), class_weight='balanced')
    model_cvs = LogisticRegression(penalty=None, C=float('inf'), class_weight='balanced')
    model_cl = LogisticRegression(penalty=None, C=float('inf'), class_weight='balanced')
    model_prl = LogisticRegression(penalty=None, C=float('inf'), class_weight='balanced')
    
    # Normal
    model.fit(X_train, y_train)
    # scores.append(model.score(X_test, y_test))
    y_pred_proba = model.predict_proba(X_test)[::,1]
    y_pred = model.predict(X_test)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    accuracy = model.score(X_test, y_test)
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()
    sensitivity = (tp)/(tp + fn)
    specificity = (tn)/(tn + fp)
    aucs.append(auc)
    accuracys.append(accuracy)
    sensitivitys.append(sensitivity)
    specificitys.append(specificity)
    
    # cvs
    model_cvs.fit(X_cvs_train, y_train)
    y_cvs_pred_proba = model_cvs.predict_proba(X_cvs_test)[::,1]
    y_cvs_pred = model_cvs.predict(X_cvs_test)
    auc_cvs = metrics.roc_auc_score(y_test, y_cvs_pred_proba)
    accuracy_cvs = model_cvs.score(X_cvs_test, y_test)
    tn_cvs, fp_cvs, fn_cvs, tp_cvs = metrics.confusion_matrix(y_test, y_cvs_pred).ravel()
    sensitivity_cvs = (tp_cvs)/(tp_cvs + fn_cvs)
    specificity_cvs = (tn_cvs)/(tn_cvs + fp_cvs)
    aucs_cvs.append(auc_cvs)
    accuracys_cvs.append(accuracy_cvs)
    sensitivitys_cvs.append(sensitivity_cvs)
    specificitys_cvs.append(specificity_cvs)
    
    # cl
    model_cl.fit(X_cl_train, y_train)
    y_cl_pred_proba = model_cl.predict_proba(X_cl_test)[::,1]
    y_cl_pred = model_cl.predict(X_cl_test)
    auc_cl = metrics.roc_auc_score(y_test, y_cl_pred_proba)
    accuracy_cl = model_cl.score(X_cl_test, y_test)
    tn_cl, fp_cl, fn_cl, tp_cl = metrics.confusion_matrix(y_test, y_cl_pred).ravel()
    sensitivity_cl = (tp_cl)/(tp_cl + fn_cl)
    specificity_cl = (tn_cl)/(tn_cl + fp_cl)
    aucs_cl.append(auc_cl)
    accuracys_cl.append(accuracy_cl)
    sensitivitys_cl.append(sensitivity_cl)
    specificitys_cl.append(specificity_cl)
    
    # prl
    model_prl.fit(X_prl_train, y_train)
    y_prl_pred_proba = model_prl.predict_proba(X_prl_test)[::,1]
    y_prl_pred = model_prl.predict(X_prl_test)
    auc_prl = metrics.roc_auc_score(y_test, y_prl_pred_proba)
    accuracy_prl = model_prl.score(X_prl_test, y_test)
    tn_prl, fp_prl, fn_prl, tp_prl = metrics.confusion_matrix(y_test, y_prl_pred).ravel()
    sensitivity_prl = (tp_prl)/(tp_prl + fn_prl)
    specificity_prl = (tn_prl)/(tn_prl + fp_prl)
    aucs_prl.append(auc_prl)
    accuracys_prl.append(accuracy_prl)
    sensitivitys_prl.append(sensitivity_prl)
    specificitys_prl.append(specificity_prl)
    
    # DeLong Tests
    model_cvs_ps.append(np.power(10, delong_roc_test(y_test, y_pred, y_cvs_pred)))
    model_cl_ps.append(np.power(10, delong_roc_test(y_test, y_pred, y_cl_pred)))
    model_prl_ps.append(np.power(10, delong_roc_test(y_test, y_pred, y_prl_pred)))
    cvs_cl_ps.append(np.power(10, delong_roc_test(y_test, y_cvs_pred, y_cl_pred)))
    cvs_prl_ps.append(np.power(10, delong_roc_test(y_test, y_cvs_pred, y_prl_pred)))
    cl_prl_ps.append(np.power(10, delong_roc_test(y_test, y_cl_pred, y_prl_pred)))
    
#%% Print results
# print(f'scores : {np.mean(scores)}')
print(f'auc : {np.mean(aucs)}')
print(f'accuracy : {np.mean(accuracys)}')
print(f'sensitivity : {np.mean(sensitivitys)}')
print(f'specificity : {np.mean(specificitys)}')

print(f'auc cvs : {np.mean(aucs_cvs)}')
print(f'accuracy cvs : {np.mean(accuracys_cvs)}')
print(f'sensitivity cvs : {np.mean(sensitivitys_cvs)}')
print(f'specificity cvs : {np.mean(specificitys_cvs)}')

print(f'auc cl : {np.mean(aucs_cl)}')
print(f'accuracy cl : {np.mean(accuracys_cl)}')
print(f'sensitivity cl : {np.mean(sensitivitys_cl)}')
print(f'specificity cl : {np.mean(specificitys_cl)}')

print(f'auc prl : {np.mean(aucs_prl)}')
print(f'accuracy prl : {np.mean(accuracys_prl)}')
print(f'sensitivity prl : {np.mean(sensitivitys_prl)}')
print(f'specificity prl : {np.mean(specificitys_prl)}')

print(f'DeLong Test:')
print(f'model ~ CVS : {model_cvs_ps} => p-value = {np.nanmean(model_cvs_ps)}')
print(f'model ~ CL : {model_cl_ps} => p-value = {np.nanmean(model_cl_ps)}')
print(f'model ~ PRL : {model_prl_ps} => p-value = {np.nanmean(model_prl_ps)}')
print(f'CVS ~ CL : {cvs_cl_ps} => p-value = {np.nanmean(cvs_cl_ps)}')
print(f'CVS ~ CL : {cvs_cl_ps} => p-value = {np.nanmean(cvs_cl_ps)}')
print(f'CL ~ PRL : {cl_prl_ps} => p-value = {np.nanmean(cl_prl_ps)}')

#%%
# =============================================================================
# Old test
# =============================================================================

#%% regression linéaire 
X = df[['% perivenular les', 'Number_CL', 'number_PRL']]
y = df['diagnosis']


# #split the dataset into training (70%) and testing (30%) sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

#instantiate the model
log_reg = LogisticRegression(penalty=None, C=float('inf'), class_weight='balanced')

#fit the model using the training data
log_reg.fit(X_train,y_train)

#define metrics
y_pred_proba = log_reg.predict_proba(X_test)[::,1]
y_pred = log_reg.predict(X_test)
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
accuracy = log_reg.score(X_test, y_test)
tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()
sensitivity = (tp)/(tp + fn)
specificity = (tn)/(tn + fp)

#%%
plt.figure()
ax = plt.axes()
#create ROC curve
plt.plot(fpr, tpr, label="AUC = "+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.title(f'ROC')
plt.show()

#%%
X_t = pd.DataFrame({'% perivenular les':[0,100], 'Number_CL':[0,10], 'number_PRL':[0,10]})
y_t = pd.DataFrame({'diagnosis':[0,1]})
#instantiate the model
test_reg = LogisticRegression()
test_reg.fit(X_t, y_t)

#fit the model using the training data
test_reg.coef_ = np.array([[0.12726271, 0.96845914, 0.8225831]])
test_reg.intercept_ = np.array([-6.29917142])


#define metrics
y_pred_proba_t = test_reg.predict_proba(X)[::,1]
y_pred_t = test_reg.predict(X)
fpr_t, tpr_t, _ = metrics.roc_curve(y, y_pred_proba_t)
auc_t = metrics.roc_auc_score(y, y_pred_proba_t)
accuracy_t = log_reg.score(X, y)
tn_t, fp_t, fn_t, tp_t = metrics.confusion_matrix(y, y_pred_t).ravel()
sensitivity_t = (tp_t)/(tp_t + fn_t)
specificity_t = (tn_t)/(tn_t + fp_t)

#%%
plt.figure()
ax = plt.axes()
#create ROC curve
plt.plot(fpr_t, tpr_t, label="AUC = "+str(auc_t))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.title(f'ROC')
plt.show()
