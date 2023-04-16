#%% -----------------------------------
import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import os as os
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#%% -----------------------------------
# get current working directory
wd = os.getcwd()

# read in entire survey
# for surface
# sl19 = pd.read_stata("~/Documents/code/csci5622/csci5622mod4/proj/data/SLHR7ADT/SLHR7AFL.DTA")
# for PC
# sl19 = pd.read_stata("C:/Users/ilyon/OneDrive - UCB-O365/Documents/code/csci5622/mod4/proj/data/SLHR7ADT/SLHR7AFL.DTA")

# generalized
sl19 = pd.read_stata(wd + "/data/SLHR7ADT/SLHR7AFL.DTA")

#%% -----------------------------------
# keep just columns of interest
cols2keep = ["hv000","hv001","hv006","hv007","hv010","hv011","hv012","hv013","hv014","hv024","hv025","hv040","hv045c","hv201","hv204","hv205","hv206","hv207","hv208","hv209","hv210","hv211","hv212","hv213","hv214","hv215","hv216","hv217","hv219","hv220","hv221","hv226","hv227","hv230a","hv237","hv241","hv243a","hv243b","hv243c","hv243d","hv243e","hv244","hv245","hv246","hv246a","hv246b","hv246c","hv246d","hv246e","hv246f","hv247","hv270","hv271","hv270a","hv271a","hml1"]
sl19keep = sl19[cols2keep]

#%% -----------------------------------
# keep just numeric variables of interest
intCols = ["hv010","hv011","hv012","hv014","hv216","hv270"]
sl19num = sl19[intCols]

# export csv
sl19num.to_csv(wd + "/data/sl19svm.csv")

# copy to prepare for cleaning
df = sl19num
df["hv270"]=pd.factorize(df["hv270"])[0]
df.describe()

#%% -----------------------------------
# convert categorical
# df.loc[df["hv245"] == "don't know", "hv245"] = 10
# df.loc[df["hv245"] == "unknown", "hv245"] = 10
# df.loc[df["hv245"] == "95 or over", "hv245"] = 950
# df["hv245"]=pd.factorize(df["hv245"])[0]
# df.loc[df["hv245"] == -1, "hv245"] = 10

#%% -----------------------------------
# scale data with mean = 0, stddev = 1
scaler = StandardScaler()
sl19scaled = scaler.fit_transform(df)

# record mean, variance in order to scale back
means = scaler.mean_
stddevs = scaler.scale_

# export csv
sl19scaledDf = pd.DataFrame(sl19scaled)
sl19scaledDf.to_csv(wd + "/data/sl19scaled.csv")

#%% -----------------------------------
# reduce columns down to 2 for clustering
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(sl19scaled)
principalDf = pd.DataFrame(data = principalComponents, columns = ['pc1', 'pc2'])
z = "hv270" # wealth index combined for urban
finalDf = pd.concat([principalDf, sl19[[z]]], axis = 1)

# plot the principal components with another variable for color
sns.set(rc={'figure.figsize':(8,10)})
sns.relplot(data=finalDf, x="pc1", y="pc2", hue=z, size=0.5).set(title="2 Principal Components of Numerical Data by Wealth Index")

#%% -----------------------------------
# split test and train data
df_train, df_test = train_test_split(df, test_size=0.2)

# remove labels
labels_test = df_test["hv270"]
df_test_nolabels = df_test.drop(["hv270"], axis=1)
labels_train = df_train["hv270"]
df_train_nolabels = df_train.drop(["hv270"], axis=1)

#%% -----------------------------------
## LINEAR KERNEL
# fit SVM model
svm_model1=LinearSVC(C=1)
svm_model1.fit(df_train_nolabels, labels_train)

# predict test data
pred_test_1 = svm_model1.predict(df_test_nolabels)

# evaluate predictions
cf_matrix_1 = confusion_matrix(labels_test, pred_test_1)
print(cf_matrix_1)
accuracy_1 = accuracy_score(pred_test_1, labels_test)
print(accuracy_1)

# plot confusion matrix
ax1 = sns.heatmap(cf_matrix_1/np.sum(cf_matrix_1), annot=True, fmt='.1%')
ax1.set_xlabel("Predicted Classes")
ax1.set_ylabel("Actual Classes")
ax1.set_title("Confusion Matrix for Linear Kernel, C=1")
ax1.text(2,6, "Accuracy: "+str(round(accuracy_1*100, 1))+"%")
# %%

#%% -----------------------------------
## RBF KERNEL
svm_model2=SVC(C=1.0, kernel='rbf', degree=3, gamma="auto")
svm_model2.fit(df_train_nolabels, labels_train)

# predict test data
pred_test_2 = svm_model1.predict(df_test_nolabels)

# evaluate predictions
cf_matrix_2 = confusion_matrix(labels_test, pred_test_2)
print(cf_matrix_2)
accuracy_2 = accuracy_score(pred_test_2, labels_test)
print(accuracy_2)

# plot confusion matrix
ax2 = sns.heatmap(cf_matrix_2/np.sum(cf_matrix_2), annot=True, fmt='.1%')
ax2.set_xlabel("Predicted Classes")
ax2.set_ylabel("Actual Classes")
ax2.set_title("Confusion Matrix for RBF Kernel, C=1")
ax2.text(2,6, "Accuracy: "+str(round(accuracy_2*100, 1))+"%")

#%% -----------------------------------
## POLYNOMIAL KERNEL
svm_model3=SVC(C=1.0, kernel='poly', degree=2, gamma="auto")
svm_model3.fit(df_train_nolabels, labels_train)

# predict test data
pred_test_3 = svm_model1.predict(df_test_nolabels)

# evaluate predictions
cf_matrix_3 = confusion_matrix(labels_test, pred_test_3)
print(cf_matrix_3)
accuracy_3 = accuracy_score(pred_test_3, labels_test)
print(accuracy_3)

# plot confusion matrix
ax3 = sns.heatmap(cf_matrix_3/np.sum(cf_matrix_3), annot=True, fmt='.1%')
ax3.set_xlabel("Predicted Classes")
ax3.set_ylabel("Actual Classes")
ax3.set_title("Confusion Matrix for Polynomial Kernel, C=1")
ax3.text(2,6, "Accuracy: "+str(round(accuracy_3*100, 1))+"%")