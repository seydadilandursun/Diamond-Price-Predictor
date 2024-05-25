import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn. linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn import metrics

data_df = pd.read_csv("diamonds.csv")
data_df.sample(20)

data_df.shape

"""Definition of Variables

* **carat (0.2-5.01):** The carat is the diamond’s physical weight measured in metric carats. One carat equals 0.20 gram and is subdivided into 100 points.
* **cut (Fair, Good, Very Good, Premium, Ideal):** The quality of the cut. The more precise the diamond is cut, the more captivating the diamond is to the eye thus of high grade.
* **color (from J (worst) to D (best)):** The colour of gem-quality diamonds occurs in many hues. In the range from colourless to light yellow or light brown. Colourless diamonds are the rarest. Other natural colours (blue, red, pink for example) are known as "fancy,” and their colour grading is different than from white colorless diamonds.
* **clarity (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best)):** Diamonds can have internal characteristics known as inclusions or external characteristics known as blemishes. Diamonds without inclusions or blemishes are rare; however, most characteristics can only be seen with magnification.
* **depth (43-79)**: It is the total depth percentage which equals to z / mean(x, y) = 2 * z / (x + y). The depth of the diamond is its height (in millimetres) measured from the culet (bottom tip) to the table (flat, top surface) as referred in the labelled diagram above.
* **table (43-95):** It is the width of the top of the diamond relative to widest point. It gives diamond stunning fire and brilliance by reflecting lights to all directions which when seen by an observer, seems lustrous.
* **price ($$326 - $18826):** It is the price of the diamond in US dollars. **It is our very target column in the dataset.**
* **x (0 - 10.74):** Length of the diamond (in mm)
* **y (0 - 58.9):** Width of the diamond (in mm)
* **z (0 - 31.8):** Depth of the diamond (in mm)

Checking for missing values & categorical variables
"""

# Checking for missing values and categorical variables in the dataset
data_df.info()

#Similar with the code above (during the data course, we used this one)
data_df.dtypes

#There is no missing data. So, we don't fill any blanks.
data_df.isnull().sum().sum()

"""Evaluating categorical features

"""

plt.figure(figsize=(10,8))
cols = ['blue','pink','yellow','green','red']
ax = sns.violinplot(x="cut",y="price", data=data_df, palette=cols,scale= "count")
ax.set_title("Diamond Cut for Price")
ax.set_ylabel("Price")
ax.set_xlabel("Cut")
plt.show()

plt.figure(figsize=(12,8))
ax = sns.violinplot(x="color",y="price", data=data_df, palette=cols,scale= "count")
ax.set_title("Diamond Colors for Price")
ax.set_ylabel("Price")
ax.set_xlabel("Color")
plt.show()

plt.figure(figsize=(13,8))
ax = sns.violinplot(x="clarity",y="price", data=data_df, palette=cols,scale= "count")
ax.set_title("Diamond Clarity for Price")
ax.set_ylabel("Price")
ax.set_xlabel("Clarity")
plt.show()

#Summary of the data
data_df.describe()

#Studying two things at once using pairplot to see how they're related.
ax = sns.pairplot(data_df, hue= "cut", palette = cols)

"""Checking for Potential Outliers



"""

lm = sns.lmplot(x="price", y="y", data=data_df, scatter_kws={"color": 'blue'}, line_kws={"color": 'pink'})
plt.title("Line Plot on Price vs 'y'")
plt.show()

lm = sns.lmplot(x="price", y="z", data=data_df, scatter_kws={"color": 'yellow'}, line_kws={"color": 'pink'})
plt.title("Line Plot on Price vs 'z'")
plt.show()

lm = sns.lmplot(x="price", y="depth", data=data_df, scatter_kws={"color": 'brown'}, line_kws={"color": 'pink'})
plt.title("Line Plot on Price vs 'depth'")
plt.show()

lm = sns.lmplot(x="price", y="table", data=data_df, scatter_kws={"color":'purple'}, line_kws={"color": 'pink'})
plt.title("Line Plot on Price vs 'Table'")
plt.show()

"""### **<span style="color:#682F2F;"><center>Data Cleaning</center></span>**"""

# Removing the "Unnamed" column
data_df = data_df.drop(["Unnamed: 0"], axis=1)
data_df.shape

# Removing the datapoints having min 0 value in x, y or z features
data_df = data_df.drop(data_df[data_df["x"]==0].index)
data_df = data_df.drop(data_df[data_df["y"]==0].index)
data_df = data_df.drop(data_df[data_df["z"]==0].index)
data_df.shape

"""
Removing Outliers"""

# Dropping the outliers (since we have huge dataset) by defining appropriate measures across features

data_df = data_df[(data_df["depth"]<75)&(data_df["depth"]>45)]
data_df = data_df[(data_df["table"]<80)&(data_df["table"]>40)]
data_df = data_df[(data_df["x"]<40)]
data_df = data_df[(data_df["y"]<40)]
data_df = data_df[(data_df["z"]<40)&(data_df["z"]>2)]
data_df.shape

"""Encoding Variables



"""

# Making a copy to keep original data
data1 = data_df.copy()

# Applying label encoder to columns with categorical data
columns = ['cut','color','clarity']
label_encoder = LabelEncoder()
for col in columns:
    data1[col] = label_encoder.fit_transform(data1[col])
data1.describe()

# Examining correlation matrix using heatmap
cmap = sns.diverging_palette(205, 133, 63, as_cmap=True)
cols = (['red', 'blue', 'yellow', 'green', 'pink', 'brown'])
corrmat= data1.corr()
f, ax = plt.subplots(figsize=(15,12))
sns.heatmap(corrmat,cmap=cols,annot=True)

"""Regression Models

"""

# Defining X and y variables
X= data1.drop(["price"],axis =1)
y= data1["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20, random_state=25)

# Building pipelins of standard scaler and model for varios regressors.

pipeline_lr=Pipeline([("scalar1",StandardScaler()),
                     ("lr",LinearRegression())])

pipeline_lasso=Pipeline([("scalar2", StandardScaler()),
                      ("lasso",Lasso())])

pipeline_dt=Pipeline([("scalar3",StandardScaler()),
                     ("dt",DecisionTreeRegressor())])

pipeline_rf=Pipeline([("scalar4",StandardScaler()),
                     ("rf",RandomForestRegressor())])


pipeline_kn=Pipeline([("scalar5",StandardScaler()),
                     ("kn",KNeighborsRegressor())])


pipeline_xgb=Pipeline([("scalar6",StandardScaler()),
                     ("xgb",XGBRegressor())])

# List of all the pipelines
pipelines = [pipeline_lr, pipeline_lasso, pipeline_dt, pipeline_rf, pipeline_kn, pipeline_xgb]

# Dictionary of pipelines and model types for ease of reference
pipeline_dict = {0: "LinearRegression", 1: "Lasso", 2: "DecisionTree", 3: "RandomForest",4: "KNeighbors", 5: "XGBRegressor"}

# Fit the pipelines
for pipe in pipelines:
    pipe.fit(X_train, y_train)

cv_results_rms = []
for i, model in enumerate(pipelines):
    cv_score = cross_val_score(model, X_train,y_train,scoring="neg_root_mean_squared_error", cv=12)
    cv_results_rms.append(cv_score)
    print("%s: %f " % (pipeline_dict[i], -1 * cv_score.mean()))

print("Linear Regression")
pred = pipeline_lr.predict(X_test)
print("R^2",metrics.r2_score(y_test, pred))
print("MSE",metrics.mean_squared_error(y_test, pred))
print("MAE:",metrics.mean_absolute_error(y_test, pred))
print("Adjusted R^2:",1 - (1-metrics.r2_score(y_test, pred))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))

print("Lasso")
pred = pipeline_lasso.predict(X_test)
print("R^2",metrics.r2_score(y_test, pred))
print("MSE",metrics.mean_squared_error(y_test, pred))
print("MAE:",metrics.mean_absolute_error(y_test, pred))
print("Adjusted R^2:",1 - (1-metrics.r2_score(y_test, pred))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))

print("Decision Tree Regressor")
pred = pipeline_dt.predict(X_test)
print("R^2",metrics.r2_score(y_test, pred))
print("MSE",metrics.mean_squared_error(y_test, pred))
print("MAE:",metrics.mean_absolute_error(y_test, pred))
print("Adjusted R^2:",1 - (1-metrics.r2_score(y_test, pred))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))

print("Random Forest Regressor")
pred = pipeline_rf.predict(X_test)
print("R^2",metrics.r2_score(y_test, pred))
print("MSE",metrics.mean_squared_error(y_test, pred))
print("MAE:",metrics.mean_absolute_error(y_test, pred))
print("Adjusted R^2:",1 - (1-metrics.r2_score(y_test, pred))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))

print("K-Neighbor Regressor")
pred = pipeline_kn.predict(X_test)
print("R^2",metrics.r2_score(y_test, pred))
print("MSE",metrics.mean_squared_error(y_test, pred))
print("MAE:",metrics.mean_absolute_error(y_test, pred))
print("Adjusted R^2:",1 - (1-metrics.r2_score(y_test, pred))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))

print("Model prediction on test data with XGBClassifier which gave us the least RMSE")
pred = pipeline_xgb.predict(X_test)
print("R^2",metrics.r2_score(y_test, pred))
print("MSE",metrics.mean_squared_error(y_test, pred))
print("MAE:",metrics.mean_absolute_error(y_test, pred))
print("Adjusted R^2:",1 - (1-metrics.r2_score(y_test, pred))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))
