#to Get Reproducible Results with Keras #https://machinelearningmastery.com/reproducible-results-neural-networks-keras/
from numpy.random import seed
seed(1)
import tensorflow
tensorflow.random.set_seed(1234)
####################################################

#https://www.dataquest.io/blog/learning-curves-machine-learning/

import pandas as pd
import numpy as np
df = pd.read_csv("dataset.csv")
df = df.dropna() # To drop Null values 

print(df.shape)

print("&&&&&&&&")

print(df.head())




from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler #For feature normalization

scaler = MinMaxScaler()


features = ['Efficiency','Specificity','BS_Length','Distance_exon_BS (D)'] #'K_Avg_Fold_change', 'K_Median_Fold_change'
target = 'K_Avg_Fold_change'
#target='K_Median_Fold_change'
###########################################
X = df[features]

#insert onehot encoding of reference-kmer
#Onehot=pd.get_dummies(df['sgRNA_sequence'], prefix='sgRNA_sequence')
#X= pd.concat([X,Onehot],axis=1)


y = df[target]

a,b=X.shape
print("#############",X.shape)
print(y.shape)


#print(X.head())

#scale training data
X= scaler.fit_transform(X)
print(",,,,,,,,",X.shape)


#train, test = train_test_split(df, test_size=0.2)   

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


###################Build NN model
'''
#from keras.models import Sequential
#from keras.layers import Dense 
#from keras.optimizers import SGD
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
# Evaluate the model: Model Accuracy, how often is the classifier correct
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import classification_report #for classifier evaluation
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score # for printing AUC
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
'''
############################################
#old stuff
###########
# from https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/
#model = Sequential()
#model.add(Dense(12, input_dim=b, activation='relu'))
#model.add(Dense(8, activation='relu'))
#####model.add(Dense(1, activation='sigmoid'))
#########################################################################
from pandas import read_csv

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

#from keras.models import Sequential
#from keras.layers import Dense
#from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
# load dataset
#dataframe = read_csv("housing.csv", delim_whitespace=True, header=None)
#dataset = dataframe.values
# split into input (X) and output (Y) variables
X = X
Y = y
# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(13, input_dim=b, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model
# evaluate model
estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)
model=estimator
kfold = KFold(n_splits=10)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))

'''
#########################

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
#plt.title('Learning curves for Linear regression model', fontsize = 18, y = 1.03)
plt.title('Learning curves for Random forest regression model', fontsize = 18, y = 1.03)
#plt.title('Learning curves for Support vector regression model', fontsize = 18, y = 1.03)
plt.legend()
#plt.ylim(0,40)
plt.savefig('NNR_cv_LC.png',dpi=300)
plt.savefig('NNR_cv_LC.svg',dpi=300)
plt.close()

########################
'''

