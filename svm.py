# %% -----------------------------------------------------------------
# IMPORT NECESSARY LIBRARIES
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
import os as os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# get current working directory
wd = os.getcwd()

# %% -----------------------------------------------------------------
# READ IN SURVEY DATA
sl19 = pd.read_stata(wd + "/data/SLHR7ADT/SLHR7AFL.DTA")

# %% -----------------------------------------------------------------
# FILTER DOWN
# keep just columns of interest and rexport
cols2keep = ["hv000", "hv001", "hv006", "hv007", "hv010", "hv011", "hv012", "hv013", "hv014", "hv024", "hv025", "hv040", "hv045c", "hv201", "hv204", "hv205", "hv206", "hv207", "hv208", "hv209", "hv210", "hv211", "hv212", "hv213", "hv214", "hv215", "hv216", "hv217",
             "hv219", "hv220", "hv221", "hv226", "hv227", "hv230a", "hv237", "hv241", "hv243a", "hv243b", "hv243c", "hv243d", "hv243e", "hv244", "hv245", "hv246", "hv246a", "hv246b", "hv246c", "hv246d", "hv246e", "hv246f", "hv247", "hv270", "hv271", "hv270a", "hv271a", "hml1"]
sl19keep = sl19[cols2keep]
# export csv
sl19keep.to_csv(wd + "/exports/sl19keep.csv", index=False)
print("sl19keep.csv exported!")
# %% -----------------------------------------------------------------
# FILTER AGAIN TO JUST SVM VARIABLES
sl19keep = pd.read_csv(wd + "/exports/sl19keep.csv")
df = sl19keep

# keep just numeric variables of interest
cols2convert = [
    "hv243a",
    "hv244",
    "hv209",
    "hv208",
    "hv243b",
    "hv206",
    "hv207"
]
intCols = ["hv270", "hv010", "hv011", "hv012", "hv014", "hv216"]
allCols = np.concatenate((intCols, cols2convert))
df = df[allCols]

# export csv
sl19svm = df
sl19svm.to_csv(wd + "/exports/sl19svm.csv", index=False)
print("sl19svm.csv exported!")

# %% -----------------------------------------------------------------
# ENCODE CATEGORICAL VARIABLES
sl19svm = pd.read_csv(wd + "/exports/sl19svm.csv")
df = sl19svm

# convert labels from words to factors
df["hv270"] = pd.factorize(df["hv270"])[0]

# convert data values from words to integers
df = df.replace("yes", 1)
df = df.replace("no", 0)
df = df.replace("none", 0)
df = df.replace("unknown", 0)
df = df.replace("urban", 1)
df = df.replace("rural", 0)
# df = df.replace("95 or more", 95)

# convert column types to numerical columns
for c in cols2convert:
    df[c] = pd.to_numeric(df[c])

# export csv
sl19num = df
sl19num.to_csv(wd + "/exports/sl19num.csv", index=False)
print("sl19num.csv exported!")

# %% ------------------------------------------------------------------
# SCALE DATASET FOR PCA
sl19num = pd.read_csv(wd + "/exports/sl19num.csv")
df = sl19num

# remove label
labels = df["hv270"]
df = df.drop(["hv270"], axis=1)

# export csv
df = pd.DataFrame(StandardScaler().fit_transform(df))
df["hv270"] = labels
sl19scaled = df
sl19scaled.to_csv(wd + "/exports/sl19scaled.csv", index=False)
print("sl19scaled.csv exported!")

# %% -----------------------------------------------------------------
# RUN & VISUALIZE PCA
sl19scaled = pd.read_csv(wd + "/exports/sl19scaled.csv")
df = sl19scaled
z = "hv270"
labels = df[z]
df = df.drop([z], axis=1)

# reduce down to 2 principal components
principal_components = PCA(n_components=2).fit_transform(df)
df_pca = pd.DataFrame(data=principal_components, columns=['pc1', 'pc2'])
df_final = pd.concat([df_pca, labels], axis=1)

# plot the principal components with another variable for color
sns.set(rc={'figure.figsize': (8, 10)})
sns.relplot(data=df_final, x="pc1", y="pc2", hue=z, size=0.5).set(
    title="2 Principal Components of Numerical Data by Wealth Index")

# %% -----------------------------------------------------------------
# SPLIT TRAIN AND TEST DATA
sl19scaled = pd.read_csv(wd + "/exports/sl19scaled.csv")
df = sl19scaled

# split
df_train, df_test = train_test_split(df, test_size=0.2)

# remove labels
labels_test = df_test["hv270"]
df_test_nolabels = df_test.drop(["hv270"], axis=1)
labels_train = df_train["hv270"]
df_train_nolabels = df_train.drop(["hv270"], axis=1)

# %% -----------------------------------------------------------------
# LINEAR KERNEL
# fit SVM model
c1 = 1
svm_model1 = LinearSVC(C=c1, max_iter=1000)
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
ax1.set_title("Confusion Matrix for Linear Kernel, C="+str(c1))
ax1.text(2, 6, "Accuracy: "+str(round(accuracy_1*100, 1))+"%")
# %% -----------------------------------------------------------------
# RBF KERNEL
c2 = 10
svm_model2 = SVC(C=c2, kernel='rbf', degree=3, gamma="auto", max_iter=100000)
svm_model2.fit(df_train_nolabels, labels_train)

# predict test data
pred_test_2 = svm_model2.predict(df_test_nolabels)

# evaluate predictions
cf_matrix_2 = confusion_matrix(labels_test, pred_test_2)
print(cf_matrix_2)
accuracy_2 = accuracy_score(pred_test_2, labels_test)
print(accuracy_2)

# plot confusion matrix
ax2 = sns.heatmap(cf_matrix_2/np.sum(cf_matrix_2), annot=True, fmt='.1%')
ax2.set_xlabel("Predicted Classes")
ax2.set_ylabel("Actual Classes")
ax2.set_title("Confusion Matrix for RBF Kernel, C="+str(c2))
ax2.text(2, 6, "Accuracy: "+str(round(accuracy_2*100, 1))+"%")

# %%

# %% -----------------------------------------------------------------
# POLYNOMIAL KERNEL
c3 = 1
svm_model3 = SVC(C=c3, kernel='poly', degree=3, gamma="auto", max_iter=10000)
svm_model3.fit(df_train_nolabels, labels_train)

# predict test data
pred_test_3 = svm_model3.predict(df_test_nolabels)

# evaluate predictions
cf_matrix_3 = confusion_matrix(labels_test, pred_test_3)
print(cf_matrix_3)
accuracy_3 = accuracy_score(pred_test_3, labels_test)
print(accuracy_3)

# plot confusion matrix
ax3 = sns.heatmap(cf_matrix_3/np.sum(cf_matrix_3), annot=True, fmt='.1%')
ax3.set_xlabel("Predicted Classes")
ax3.set_ylabel("Actual Classes")
ax3.set_title("Confusion Matrix for Polynomial Kernel 3rd Deg, C="+str(c3))
ax3.text(2, 6, "Accuracy: "+str(round(accuracy_3*100, 1))+"%")
# %%
