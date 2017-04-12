###################################################
# step 1 - import libraries and data
###################################################

import pandas as pd
import re
import os
from sklearn import model_selection as ms
from sklearn import linear_model as lm
from sklearn import preprocessing as p
from sklearn import metrics, cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import cross_val_score
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

os.chdir('/Users/ntmill/Library/Mobile Documents/com~apple~CloudDocs/Github/titanic')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

###################################################
# step 2 - feature engineering
###################################################

# parse title from name
def findTitle(name):
    match = re.search("(Dr|Mrs?|Ms|Miss|Master|Rev|Capt|Mlle|Col|Major|Sir|Jonkheer|Lady|the Countess|Mme|Don)\\.",name)
    if match:
        title = match.group(0)
        if (title == 'Mr.'):
            title = 'untitled_male'
        if (title == 'Master.' or title == 'Dr.' or title == 'Rev.' or title == 'Sir.' or title == 'Col.' or title == 'Jonkheer.' or title == 'Capt.' or title == 'Don.' or title == 'Major.'):
            title = 'titled_maled'
        if (title == 'Miss.' or title == 'Ms.' or title == 'Mlle.' or title == 'Mme.'):
            title = 'untitled_female'
        if (title == 'Mrs.' or title == 'Lady.' or title == 'the Countess.'):
            title = 'titled_female'
        return title
    else:
        return "other"

# define class
def Class(Pclass):
    if Pclass == 1:
        Class = 'upper'
    if Pclass == 2:
        Class = 'middle'
    if Pclass == 3:
        Class = 'lower'
    return Class


# update training dataset
train['Title'] = train['Name'].apply(findTitle)
train['Class'] = train['Pclass'].apply(Class)

# impute na values with average ages by title and class
age_mean = train.groupby(['Title','Class'])['Age'].mean()
train = train.set_index(['Title','Class'])
train['Age'] = train['Age'].fillna(age_mean)
train = train.reset_index()

# some more variables that sound fun
train['Fare_x_age'] = train['Age'] * train['Fare']
train['is_fam'] = 0     
train.loc[train['SibSp'] > 0, 'is_fam'] = 1
train.loc[train['Parch'] > 0, 'is_fam'] = 1
train['tot_fam'] = train['SibSp'] + train['Parch']
      
# create dummies for training dataset
train_gender_dummies = pd.get_dummies(train['Sex'], dummy_na=True)
train_gender_dummies.columns = ['gender_' + str(col) for col in train_gender_dummies.columns]
train_embarked_dummies = pd.get_dummies(train['Embarked'], dummy_na=True)
train_embarked_dummies.columns = ['embarked_' + str(col) for col in train_embarked_dummies.columns]
train_title_dummies = pd.get_dummies(train['Title'], dummy_na=True)
train_title_dummies.columns = ['title_' + str(col) for col in train_title_dummies.columns]
train_class_dummies = pd.get_dummies(train['Class'])
train_class_dummies.columns = ['class_' + str(col) for col in train_class_dummies.columns]


x_orig_train = train[['Age','SibSp','Parch','is_fam','tot_fam','Fare','Fare_x_age']]
x_orig_train = pd.concat([x_orig_train, train_gender_dummies, train_embarked_dummies, train_class_dummies, train_title_dummies], axis=1)
colnames = x_orig_train.columns
imp = p.Imputer(missing_values='NaN', strategy='mean')
x_orig_train = imp.fit_transform(x_orig_train)
x_orig_train = pd.DataFrame(x_orig_train)
x_orig_train.columns = colnames
x_orig_train = pd.DataFrame(x_orig_train)
y_orig_train = train['Survived']

# udpate test dataset
test["Title"] = test["Name"].apply(findTitle)
test['Class'] = test['Pclass'].apply(Class)

# impute na values with average ages by title and class
age_mean_test = test.groupby(['Title','Class'])['Age'].mean()
test = test.set_index(['Title','Class'])
test['Age'] = test['Age'].fillna(age_mean_test)
test = test.reset_index()

# some extras
test['Fare_x_age'] = test['Age'] * test['Fare']
test['is_fam'] = 0     
test.loc[test['SibSp'] > 0, 'is_fam'] = 1
test.loc[test['Parch'] > 0, 'is_fam'] = 1
test['tot_fam'] = test['SibSp'] + test['Parch']

# create dummies for test dataset
test_gender_dummies = pd.get_dummies(test['Sex'], dummy_na=True)
test_gender_dummies.columns = ['gender_' + str(col) for col in test_gender_dummies.columns]
test_embarked_dummies = pd.get_dummies(test['Embarked'], dummy_na=True)
test_embarked_dummies.columns = ['embarked_' + str(col) for col in test_embarked_dummies.columns]
test_title_dummies = pd.get_dummies(test['Title'], dummy_na=True)
test_title_dummies.columns = ['title_' + str(col) for col in test_title_dummies.columns]
test_class_dummies = pd.get_dummies(test['Class'])
test_class_dummies.columns = ['class_' + str(col) for col in test_class_dummies.columns]

x_submit = test[['Age','SibSp','Parch','is_fam','tot_fam','Fare','Fare_x_age']]
x_submit = pd.concat([x_submit, test_gender_dummies, test_embarked_dummies, test_class_dummies, test_title_dummies], axis=1)
x_submit_colnames = x_submit.columns
x_orig_submit = imp.fit_transform(x_submit)
x_orig_submit = pd.DataFrame(x_orig_submit)
x_orig_submit.columns = x_submit_colnames
x_orig_submit = x_orig_submit.drop('title_other', axis = 1)

###################################################
# step 3 - feature reduction
###################################################

# feature scaling
min_max_scaler = p.MinMaxScaler()
x_trans_train = pd.DataFrame(min_max_scaler.fit_transform(x_orig_train))
colnames = x_orig_train.columns
x_trans_train.columns = colnames

x_trans_submit = pd.DataFrame(min_max_scaler.fit_transform(x_orig_submit))
colnames_new = x_trans_submit.columns
x_trans_submit.columns = colnames
x_trans_submit = x_trans_submit.drop('title_other', axis = 1)


# random forest for feature importance
rf = RandomForestClassifier(n_estimators=500, max_features=2, min_samples_leaf=5, random_state=201, n_jobs=-1)
rf_train = rf.fit(X = x_orig_train, y = y_orig_train)
importance = rf_train.feature_importances_
importance = pd.DataFrame(importance, index=pd.DataFrame(x_orig_train).columns, columns=["Importance"])
rf_vars = pd.DataFrame(importance[importance['Importance'] >= 0.035])
rf_vars = rf_vars.index

importance.plot.bar()

###################################################
# step 4 - test train split
###################################################

X_train_orig, X_valid_orig, y_train_orig, y_valid_orig = ms.train_test_split(x_orig_train, y_orig_train, test_size=0.25, random_state=201)
X_train_trans, X_valid_trans, y_train_trans, y_valid_trans = ms.train_test_split(x_trans_train, y_orig_train, test_size=0.25, random_state=201)

###################################################
# step 6 - logistic regression
###################################################

# main effects regression model
log = lm.LogisticRegression()
log_main = log.fit(X_train_orig, y = y_train_orig)
log_main_predict = cross_validation.cross_val_predict(log_main, X_train_orig, y_train_orig, cv=10)
metrics.accuracy_score(y_train_orig, log_main_predict)
log_main_preds = log_main.predict(X_valid_orig)
log_main_preds = pd.DataFrame(log_main_preds)

# using only random forest variables
log_rfvars = log.fit(X_train_orig[rf_vars], y = y_train_orig)
log_rfvars_predict = cross_validation.cross_val_predict(log, x_orig_train[rf_vars], y_orig_train, cv=10)
metrics.accuracy_score(y_orig_train, log_rfvars_predict)

###################################################
# step 7 - random forest
###################################################

params_rf = {'max_depth': (2,3,4),
            'max_features': ['sqrt','log2'],
            'bootstrap': [True, False],
            'criterion': ['gini','entropy']}
            
rf_orig = RandomForestClassifier(n_estimators=500, max_features='sqrt', min_samples_leaf=5, random_state=201, n_jobs=-1)
rf_grid_search = GridSearchCV(rf_orig, param_grid=params_rf, cv = 5)
rf_grid_search.fit(x_orig_train, y_orig_train)
rf_grid_search.best_params_
rf_grid = RandomForestClassifier(bootstrap=False, criterion='gini', max_depth=4, max_features='sqrt')

rf_grid_search.fit(x_orig_train[rf_vars], y_orig_train)
rf_grid_search.best_params_

rf_orig_scores = cross_val_score(rf_orig, x_orig_train, y_orig_train, cv=5)
rf_grid_scores = cross_val_score(rf_grid, x_orig_train, y_orig_train, cv=5)
rf_orig_scores_trans = cross_val_score(rf_orig, x_trans_train, y_orig_train, cv=5)
rf_grid_scores_trans = cross_val_score(rf_grid, x_trans_train, y_orig_train, cv=5)

rf_orig_scores.mean()
rf_grid_scores.mean()
rf_orig_scores_trans.mean()
rf_grid_scores_trans.mean()

# select rf_orig
rf_orig_fit = rf_orig.fit(X_train_orig, y = y_train_orig)
rf_orig_preds = rf_orig.predict(X_valid_orig)
rf_orig_preds = pd.DataFrame(rf_orig_preds)
metrics.accuracy_score(rf_orig_preds, y_valid_orig)

rf_grid_fit = rf_grid.fit(X_train_orig, y = y_train_orig)
rf_grid_preds = rf_orig.predict(X_valid_orig)
rf_orig_preds = pd.DataFrame(rf_grid_preds)
metrics.accuracy_score(rf_grid_preds, y_valid_orig)

###################################################
# step 8 - xgboost
###################################################

gbm = xgb.XGBClassifier()
gbm_params = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth':range(3,10,2),
    'min_child_weight':range(1,6,2)}
 
xgb_grid = GridSearchCV(gbm, gbm_params, scoring='roc_auc', cv=5, verbose=10,n_jobs=-1)
xgb_grid.fit(x_trans_train, y_orig_train)
xgb_scores = cross_val_score(xgb_grid.best_estimator_, x_trans_train, y_orig_train, cv=5)
xgb_scores.mean()

xgb_preds = xgb_grid.predict(X_valid_trans)
metrics.accuracy_score(xgb_preds, y_valid_orig)

###################################################
# step 9 - neural network
###################################################

nn = MLPClassifier()
nn_params = {'solver': ['lbfgs','adam'],
            'learning_rate': ['constant'],
            'hidden_layer_sizes': (6,8,10,(4,2),(5,3))
}

nn_grid = GridSearchCV(nn, nn_params, cv=5, scoring='roc_auc')
nn_grid.fit(x_trans_train, y_orig_train)
nn_grid.best_estimator_
nn_scores = cross_val_score(nn_grid.best_estimator_, x_trans_train, y_orig_train, cv=5)
nn_scores.mean()

nn_preds = pd.DataFrame(nn_grid.predict(X_valid_trans))
metrics.accuracy_score(nn_preds, y_valid_orig)

###################################################
# step 10 - ensemble nn
###################################################

log_ens = pd.DataFrame(log_main_preds)
rf_ens = pd.DataFrame(rf_orig_preds)
xgb_ens = pd.DataFrame(xgb_preds)
nn_ens = pd.DataFrame(nn_preds)
tot_ens = pd.concat([log_ens, rf_ens, xgb_ens, nn_ens], axis=1)
col_ens = ['log','rf','xgb','nn']
tot_ens.columns = col_ens

# combine these into a dataframe, then build the ensemble nn prediction
nn_ens = MLPClassifier()
nn_ens_params = {'solver': ['lbfgs','adam'],
            'learning_rate': ['constant'],
            'hidden_layer_sizes': (1,2,3,(2,1),(3,1))
}
nn_ens_grid = GridSearchCV(nn_ens, nn_ens_params, cv=5, scoring='roc_auc')
nn_ens_grid.fit(tot_ens, y_valid_orig)
nn_ens_grid.best_estimator_
nn_scores = cross_val_score(nn_ens_grid.best_estimator_, tot_ens, y_valid_orig, cv=5)
nn_scores.mean()

nn_ens_preds = nn_ens_grid.predict(tot_ens)
metrics.accuracy_score(nn_ens_preds, y_valid_orig)

nn_ens_preds = pd.DataFrame(nn_ens_preds)
nn_ens_preds.describe()
tot_ens.describe()
###################################################
# step 11 - compile preds for submission file
###################################################

# compile all model predictions
log_fin = pd.DataFrame(log_main.predict(x_orig_submit)) # need to get back to this
rf_fin = pd.DataFrame(rf_orig.predict(x_orig_submit))
xgb_fin = pd.DataFrame(xgb_grid.predict(x_trans_submit))
nn_fin = pd.DataFrame(nn_grid.predict(x_trans_submit))
ens_fin = pd.concat([log_fin, rf_fin, xgb_fin, nn_fin], axis=1) # an issue with log_fin so don't use


# export random forest and xgb for final predictions
rf_fin_submit = pd.DataFrame(pd.concat([test['PassengerId'], rf_fin], axis=1))
rf_fin_submit.columns = ['PassengerId','Survived']
rf_fin_submit.to_csv('submit_rf_final.csv', index = False)

xgb_fin_submit = pd.DataFrame(pd.concat([test['PassengerId'], xgb_fin], axis=1))
xgb_fin_submit.columns = ['PassengerId','Survived']
xgb_fin_submit.to_csv('submit_xgb_final.csv', index = False)

##### NOT USING IN FINAL VERSION
nn_ens_fin = pd.DataFrame(nn_ens_grid.predict(ens_fin))
ens_fin = pd.concat([ens_fin, nn_ens_fin], axis=1)
ens_fin.columns = ['log','rf','xgb','nn','ens']

nn_ens_fin_submit = pd.DataFrame(pd.concat([test['PassengerId'], ens_fin['ens']], axis=1))
nn_ens_fin_submit.columns = ['PassengerId','Survived']
nn_ens_fin_submit.to_csv('submit_ens.csv', index = False)

