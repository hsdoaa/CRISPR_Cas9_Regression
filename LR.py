#https://www.dataquest.io/blog/learning-curves-machine-learning/

import pandas as pd
import numpy as np
df = pd.read_csv("dataset.csv")
df = df.dropna() # To drop Null values 

print(df.shape)

print("&&&&&&&&")

print(df.head())
'''
electricity = pd.read_excel('Folds5x2_pp.xlsx')
print(electricity.info())
electricity.head(3)
'''



from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler #For feature normalization

scaler = MinMaxScaler()


estimator = LinearRegression()
#estimator =RandomForestRegressor()
#estimator=SVR()#SVR(C=1.0, epsilon=0.2)
#features = ['Efficiency','Specificity','BS_Length','Distance_exon_BS (D)'] #'K_Avg_Fold_change', 'K_Median_Fold_change'
#features = ['Efficiency']
#features = ['Specificity']
#features = ['BS_Length']
features =['Distance_exon_BS (D)']

target = 'K_Avg_Fold_change'
###########################################
X = df[features]
y = df[target]

print("#############",X.shape)
print(y.shape)


#print(X.head())

#scale training data
#X= scaler.fit_transform(X)
print(",,,,,,,,",X.shape)


#train, test = train_test_split(df, test_size=0.2)   

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



model=estimator

model.fit(X_train, y_train)
print(model)
expected = y_test
predicted= model.predict(X_test)

prediction_results= cross_val_predict(model,X_test,predicted)
mse = np.mean((predicted-expected)**2)
print("###",mse)
#print(model.intercept_, model.coef_, mse)
print(model.score(X_train, y_train))

####################################################
from sklearn.metrics import mean_squared_error
from sklearn  import metrics
from math import sqrt

rmse = sqrt(mean_squared_error(y_test,predicted))
print("$$$",mse)
print("The linear regression score is {}".format(model.score(X_train,y_train)))
print("The linear regression score is {}".format(model.score(X_test,y_test)))
print("The RMSE is {}".format(rmse))
#print("The RMSE of the training set is {}".format(np.sqrt(metrics.mean_squared_error(y_train,X_train))))
#print("The MAE is {}".format(metrics.mean_absolute_error(y_test,predicted)))
print("The MSE is {}".format(metrics.mean_squared_error(y_test,predicted)))

#######################################################
#Plot learning curve
train_sizes = [1, 500, 2500, 10000, 25000, 34204]
train_sizes, train_scores, validation_scores = learning_curve(
estimator = estimator,
X = df[features],
y = df[target], train_sizes = train_sizes, cv = 5,
scoring = 'neg_mean_squared_error')

print('Training scores:\n\n', train_scores)
print('\n', '-' * 70) # separator to make the output easy to read
print('\nValidation scores:\n\n', validation_scores)

train_scores_mean = -train_scores.mean(axis = 1)
validation_scores_mean = -validation_scores.mean(axis = 1)
print('Mean training scores\n\n', pd.Series(train_scores_mean, index = train_sizes))
print('\n', '-' * 20) # separator
print('\nMean validation scores\n\n',pd.Series(validation_scores_mean, index = train_sizes))


plt.style.use('seaborn')
plt.plot(train_sizes, train_scores_mean, label = 'Training error')
plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')
plt.ylabel('MSE', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Learning curves for Linear regression model', fontsize = 18, y = 1.03)
#plt.title('Learning curves for Random forest regression model', fontsize = 18, y = 1.03)
#plt.title('Learning curves for Support vector regression model', fontsize = 18, y = 1.03)
plt.legend()
#plt.ylim(0,40)
########################
'''
def learning_curves(estimator, data, features, target, train_sizes, cv):
    train_sizes, train_scores, validation_scores = learning_curve(
    estimator, data[features], data[target], train_sizes =
    train_sizes, cv = cv,scoring ='neg_mean_squared_error')
    train_scores_mean = -train_scores.mean(axis = 1)
    validation_scores_mean = -validation_scores.mean(axis = 1)

    plt.plot(train_sizes, train_scores_mean, label = 'Training error')
    plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')

    plt.ylabel('MSE', fontsize = 14)
    plt.xlabel('Training set size', fontsize = 14)
    title = 'Learning curves for a ' + str(estimator).split('(')[0] + ' model'
    plt.title(title, fontsize = 18, y = 1.03)
    plt.legend()
    plt.ylim(0,40)
### Plotting the two learning curves ###
'''
plt.show()

