import pandas as pd
from io import StringIO


# Looking at missing data
#create the CSV data
csv_data = \
'''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,, 8.0
10.0,11.0,12.0'''

df = pd.read_csv(StringIO(csv_data))


# Null counts
df.isnull().sum()


### drop rows with NA's
df.dropna(axis= 0)

#### drop columns with NA's
df.dropna(axis= 1)

#### Drop rows with all missing values
df.dropna(how='all')

#### Drop rows with fewer than 4 real numbers
df.dropna(thresh=4)


### Drop missing in C column.
df.dropna(subset= ["C"])


#### input mean for missing values
from sklearn.impute import SimpleImputer
import numpy as np

imr = SimpleImputer(missing_values= np.nan, strategy= "mean")
imr.fit(df.values)
imputed_data = imr.transform(df.values)
imputed_data

# using pandas to imput
df.fillna(df.mean())

######Categorical Data

import pandas as pd

df = pd.DataFrame([['green', 'M', 10.1, 'class2'], ['red', 'L', 13.5, 'class1'], ['blue', 'XL', 15.3, 'class2']])
df.columns = ['color', 'size', 'price', 'classlabel']


#### size mapping
size_mapping = {
    "XL" : 3,
    "L" : 2,
    "M": 1
    }

df['size'] = df['size'].map(size_mapping)
df


##### nominal encoding
class_mapping = {
    label : idx for idx, label in enumerate(np.unique(df['classlabel']))
}

df['classlabel'] = df['classlabel'].map(class_mapping)
df


### one hot encoding
from sklearn.preprocessing import  LabelEncoder


X= df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
X

##### implementing one hot encoder
from sklearn.preprocessing import OneHotEncoder
X = df[['color', 'size','classlabel']].values
ohe = OneHotEncoder()
ohe.fit_transform(X[:, 0].reshape(-1, 1)).toarray()


#### applying one hot encoding to muliple columns.
from sklearn.compose import ColumnTransformer
X = df[['color', 'size','classlabel']].values
c_trans = ColumnTransformer([('onehot', OneHotEncoder(0), [0]), ('nothing', 'passthrough', [1,2])])
c_trans.fit_transform(X).astype(float)
X

###### But using pandas is much easier
pd.get_dummies(df[['price', 'color', 'size']])

##### Working with the wine data set.
df_wine = pd.read_csv('wine.data',header=None)
df_wine.columns = ['Class Label', "Alcohol", 'Malic Acid', 'Ash', 'Alcalinity of Ash', 'Magnesium', 'Total Phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color Intensity', 'Hue', 'OD280/OD315 of diluted wine.',
                   'Proline']
print("Class Labels: ", np.unique(df_wine['Class Label']))

