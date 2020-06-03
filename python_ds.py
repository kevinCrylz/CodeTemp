import pandas as pd

df = pd.read_csv('path', header = None)
df.to_csv('path')

df.head(n)
df.tail(n)
df.info()

# Change column headers
df.columns = ['', '', '', ...]
df.rename(columns={'old':'new'}, inplace=True)

# Check column data types
df.dtypes()
df['col'].astype('int')

df.describe(include='all')

df = pd.concat([df1, df2], axis=1)
df.drop('çol', axis=1, inplace=True)
df.dropna(subset=['col'], axis=0, inplace=True) # axis = 0 to drop row / 1 to drop column
df.fillna(0)

df = df.query('col != "val"')
df = df.assign(colName=(df['col']='val').astype(int))
df = df.assign(hour=df.col.dt.hour,
               day=df.col.dt.day,
               month=df.col.dt.month,
               year=df.col.dt.year)
df = df['col'].join(df)
df = df['col'].map(func)
df['col1'].replace(np.nan, df['col1'].mean())

df['col'].value_counts()

# Min-max scaling
df['scaled'] = (df['col'] - df['col'].min()) / (df['col'].max() - df['col'].min())

# Z score
df['scaled'] = (df['col'] - df['col'].mean()) / df['col'].std()

# Binning
bins = np.linspace(min(df['col']),max(df['col']),4)
gps = ['a','b','c']
df['binned'] = pd.cut(df['col'], bins, labels=gps, include_lowest=True)

# label encoding
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoded = df['col'].apply(encoder.fit_transform)

# one-hot encoding
pd.get_dummies(df['col'])

df.reset_index(drop=True, inplace=True)  # drop: previous index

# Groupby
df_pivot = df.groupby(['col1','col2'], as_index=False).mean().pivot(index='col1', columns='col2')

corr = df.corr()

# Plot
import seaborn as sns

sns.boxplot(x='col1',y='col2',data=df)
sns.regplot(x='col1',y='col2',data=df)
sns.heatmap(corr, cmap='RdBu')

plt.scatter(df['col1'], df['col2'])

plt.title('')
plt.xlabel('')
plt.ylabel('')

plt.pcolor(df_pivot, cmap='RdBu')
plt.colorbar()

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)
y_ = model.predict(X_)

# Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

Input = [('scale', StandardScaler()), ('polynomial', PolynomialFeatures(degree=2)), ('mode', LinearRegression())]
pipe = Pipeline(Input)
pipe.train(X, Y)
y_ = pipe.predict(X_)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

from sklearn.metrics import accuracy_score

accuracy_score(y_pred, y_test)