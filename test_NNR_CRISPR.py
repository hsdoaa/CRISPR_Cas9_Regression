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

############
#train, test = train_test_split(df, test_size=0.2)   

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


###################Build NN model

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
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_predict
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
print(y.head())
print(type(y))
a,b=X.shape
print("#############",X.shape)
print(y.shape)
#y=np.reshape(y, (-1,1))

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
print(scaler_x.fit(X))
xscale=scaler_x.transform(X)
#print(scaler_y.fit(y))
#yscale=scaler_y.transform(y)

####################################################

X_train, X_test, y_train, y_test = train_test_split(xscale, y) #yscale

###########################################
model = Sequential()
model.add(Dense(12, input_dim=b, kernel_initializer='normal', activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()

##################################

model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])

###########################
history = model.fit(X_train, y_train, epochs=150, batch_size=50,  verbose=1, validation_split=0.2)

#########################################

print(history.history.keys())
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

################################
#Predictions
Xnew = np.array([[40, 0, 26, 9000, 8000]])

Xnew = np.array([[40, 0, 26, 9000, 8000]])
Xnew= scaler_x.transform(Xnew)
ynew= model.predict(Xnew)
#invert normalize
ynew = scaler_y.inverse_transform(ynew) 
Xnew = scaler_x.inverse_transform(Xnew)
print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))
'''
#print(X.head())

#scale training data
X= scaler.fit_transform(X)
print(",,,,,,,,",X.shape)


#train, test = train_test_split(df, test_size=0.2)   

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

###################Build NN model

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
######################################################

# Initialising the ANN
model = Sequential()

# Adding the input layer and the first hidden layer
model.add(Dense(32, activation = 'relu', input_dim = b))

# Adding the second hidden layer
model.add(Dense(units = 32, activation = 'relu'))

# Adding the third hidden layer
model.add(Dense(units = 32, activation = 'relu'))

# Adding the output layer
model.add(Dense(units = 1))
model.compile(optimizer = 'adam',loss = 'mean_squared_error')
regressor=model.fit(X_train, y_train, batch_size = 10, epochs = 100)
y_pred = model.predict(X_test)

plt.plot(y_test, color = 'red', label = 'Real data')
plt.plot(y_pred, color = 'blue', label = 'Predicted data')
plt.title('Prediction')
plt.legend()
plt.show()

####################################################
predicted=y_pred
expected = y_test
#mse = np.mean((predicted-expected)**2)
#print("###",mse)
#print(model.intercept_, model.coef_, mse)
#print(model.score(X_train, y_train))

####################################################
from sklearn.metrics import mean_squared_error
from sklearn  import metrics
from math import sqrt

mse=(mean_squared_error(y_test,predicted))
rmse = sqrt(mean_squared_error(y_test,predicted))
print("$$$",mse)
print("@@@",rmse)
#print("The linear regression score is {}".format(model.score(X_train,y_train)))
#print("The linear regression score is {}".format(model.score(X_test,y_test)))
print("The RMSE is {}".format(rmse))
#print("The RMSE of the training set is {}".format(np.sqrt(metrics.mean_squared_error(y_train,X_train))))
#print("The MAE is {}".format(metrics.mean_absolute_error(y_test,predicted)))
print("The MSE is {}".format(metrics.mean_squared_error(y_test,predicted)))

#######################################################
#needed for plotting learning curve
train_loss = regressor.history['loss']
#val_loss   = regressor.history['val_loss']
train_acc  = regressor.history['mse']
val_acc    = regressor.history['val_accuracy']
xc         = range(50)


# make class predictions with the model
predictions = model.predict_classes(X_test)
y_pred = model.predict_classes(X_test)
y_prob = model.predict_proba(X_test)
#y_prob = y_prob[:,1]

print(classification_report(y_test, y_pred))
auc=roc_auc_score(y_test.round(),y_pred)
auc = float("{0:.3f}".format(auc))
print("AUC=",auc)

print("X 4Features+length before that was X1 features for 4 types of X featured concatenated with x")


fpr, tpr, thresholds = metrics.roc_curve(y_test, y_prob)


# Print ROC curve
plt.plot(fpr,tpr)
plt.title("ROC Curve")
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show() 

#################################################
'''