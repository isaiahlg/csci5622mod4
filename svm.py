import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import os as os

#%%
# get current working directory
wd = os.getcwd()

# read in entire survey
# for surface
# sl19 = pd.read_stata("~/Documents/code/csci5622/csci5622mod2/proj/data/SLHR7ADT/SLHR7AFL.DTA")
# for PC
# sl19 = pd.read_stata("C:/Users/ilyon/OneDrive - UCB-O365/Documents/code/csci5622/mod2/proj/data/SLHR7ADT/SLHR7AFL.DTA")

# generalized
sl19 = pd.read_stata(wd + "/data/SLHR7ADT/SLHR7AFL.DTA")

#%%
# keep just columns of interest
cols2keep = ["hv000","hv001","hv006","hv007","hv010","hv011","hv012","hv013","hv014","hv024","hv025","hv040","hv045c","hv201","hv204","hv205","hv206","hv207","hv208","hv209","hv210","hv211","hv212","hv213","hv214","hv215","hv216","hv217","hv219","hv220","hv221","hv226","hv227","hv230a","hv237","hv241","hv243a","hv243b","hv243c","hv243d","hv243e","hv244","hv245","hv246","hv246a","hv246b","hv246c","hv246d","hv246e","hv246f","hv247","hv270","hv271","hv270a","hv271a","hml1"]
sl19keep = sl19[cols2keep]

#%%
# keep just numeric variables of interest
intCols = ["hv010","hv011","hv012","hv014","hv216"]
sl19num = sl19[intCols]

# export csv
sl19num.to_csv(wd + "/data/sl19svm.csv")

# copy to prepare for cleaning
df = sl19num

#%%
# convert strings into ints
# TODO come back and pull out some of these categories into integers
# df.loc[df["hv204"] == "on premises", "hv204"] = 0   # water from house
# df.loc[df["hv245"] == "don't know", "hv245"] = 1
# df.loc[df["hv245"] == "unknown", "hv245"] = 1
# df.loc[df["hv245"] == "95 or over", "hv245"] = 95

#%%
# scale data with mean = 0, stddev = 1
scaler = StandardScaler()
sl19scaled = scaler.fit_transform(df)

# record mean, variance in order to scale back
means = scaler.mean_
stddevs = scaler.scale_

# export csv
sl19scaledDf = pd.DataFrame(sl19scaled)
sl19scaledDf.to_csv(wd + "/data/sl19scaled.csv")

#%%
# reduce columns down to 3 for clustering
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(sl19scaled)
principalDf = pd.DataFrame(data = principalComponents, columns = ['pc1', 'pc2'])
z = "hv270a" # wealth index combined for urban
finalDf = pd.concat([principalDf, sl19[[z]]], axis = 1)

# plot the principal components with another variable for color
sns.set(rc={'figure.figsize':(8,10)})
sns.relplot(data=finalDf, x="pc1", y="pc2", hue=z, size=0.5).set(title="2 Pricipal Components of Numerical Data by Wealth Index")

#%%