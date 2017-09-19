import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing, cross_validation
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
df.fillna(0, inplace=True)
df.drop('Id', 1, inplace=True)
df.convert_objects(convert_numeric=True)

Id = test['Id']
test.fillna(0, inplace=True)
test.drop('Id', 1, inplace=True)

##print(df.head().info())

def handle_non_numerical_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x=0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))
##        print(text_digit_vals)    
    return df

##sns.heatmap(data=df.corr())
##plt.show()
df = handle_non_numerical_data(df)
sns.factorplot(y='SalePrice', x='LotFrontage',data=df)
plt.show()
df.drop(['Street', 'Utilities', 'Condition2', 'RoofMatl', 'Heating'], 1, inplace=True)
test.drop(['Street', 'Utilities', 'Condition2', 'RoofMatl', 'Heating'], 1, inplace=True)

X = np.array(df.drop('SalePrice', 1))
X = preprocessing.scale(X)
y = df['SalePrice']

test = handle_non_numerical_data(test)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

clf = RandomForestRegressor()
clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))
p = clf.predict(test)
print(p)

with open('submission_file.csv', 'w') as f:
    f.write('Id,SalePrice\n')

print('File Created!')
with open('submission_file.csv', 'a') as f:
    for i in range(len(p)):
        f.write('{},{}\n'.format(Id[i], p[i]))
        













