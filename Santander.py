import numpy as np
import pandas as pd

test=pd.read_csv('test.csv')

train=pd.read_csv('train.csv')

test.columns

test['0deb4b6a8'].dtype

[col for col in test.columns if test[col].dtype == 'object']
[col for col in train.columns if train[col].dtype == 'object']

train[col].dtype for col in train.columns

[col for col in test.columns if test[test[col].isnull()].shape[0]>0]
[col for col in train.columns if train[train[col].isnull()].shape[0]>0]


test[test['20aa07010'].isnull()].shape[0]>0




X_train=train.iloc[:,2:]
Y_train=train.iloc[:,1]

X_test=test.iloc[:,1:]
Y_test=test.iloc[:,0]



col_to_norm=X_train.columns

train_norm=X_train
test_norm=X_test


from sklearn.preprocessing import StandardScaler
for col in col_to_norm:
    train_norm[col]=StandardScaler().fit_transform(X_train[col])
for col in col_to_norm:
    test_norm[col]=StandardScaler().fit_transform(X_test[col])

################################################################
##Regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
model=DecisionTreeRegressor(max_depth=60,min_samples_leaf=0.2, max_features=0.8)
cv = ShuffleSplit(n_splits=5,  random_state=0)
cross_val_score(model, train_norm, Y_train, cv=cv)

model.fit(train_norm,Y_train)

print("Classification rate for AdaBoost: ", model.score(train_norm,Y_train))
Y_test_pred=model.predict(X_test)

from pandas import DataFrame as df
result=df({'ID':Y_test,'target':Y_test_pred})

result.to_csv("submission_reg5.csv", index= False)
################################################################
##RandomForest

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
model=RandomForestRegressor(n_estimators=50,max_depth=100,min_samples_leaf=0.2, max_features='auto')
cv = ShuffleSplit(n_splits=5,  random_state=0)
cross_val_score(model, train_norm, Y_train, cv=cv)

model.fit(train_norm,Y_train)

print("Classification rate for AdaBoost: ", model.score(train_norm,Y_train))
Y_test_pred=model.predict(X_test)

from pandas import DataFrame as df
result=df({'ID':Y_test,'target':Y_test_pred})

result.to_csv("submission_rft4.csv", index= False)
#################################################################

from sklearn.ensemble import AdaBoostRegressor
ada_tree_backing=DecisionTreeRegressor(max_depth=60,min_samples_leaf=0.2, max_features=0.8)
model=AdaBoostRegressor(ada_tree_backing, learning_rate=0.1, loss='square', n_estimators=1000)
model.fit(X_train,Y_train)

Y_pred=model.predict(X_train)
Y_test_pred=model.predict(X_test)

print("Classification rate for AdaBoost: ", model.score(X_train,Y_train))

Y_test_pred[Y_test_pred<0]

for col in X_text.columns:
    if test[col].dtype!='float64':
        print(col)
    
from pandas import DataFrame as df
result=df({'ID':Y_test,'target':Y_test_pred})

result.to_csv("submission_ada.csv", index= False)



import xgboost as xgb
from xgboost import XGBRegressor    
xgb1 = XGBRegressor(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=10,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.5,
 #objective= 'multi:softmax',
 nthread=4,
 #scale_pos_weight=1,
 seed=27)

xgb1.fit(X_train,Y_train)

Y_pred = xgb1.predict(X_train)
Y_test_pred=xgb1.predict(X_test)
print("Classification rate for AdaBoost: ", xgb1.score(X_train,Y_train))

from pandas import DataFrame as df
result=df({'ID':Y_test,'target':Y_test_pred})

result.to_csv("submission_xgb.csv", index= False)


result[result.target<0]
