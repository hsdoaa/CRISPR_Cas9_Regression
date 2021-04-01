# CRISPR_Cas9_Regression

These are regression predictors that predict the average fold change in Crisper-cas9 dataset. Those models include:

Support vector regression (SVR)

neural network regression (NNR)

Random forest regression (RFR)

linear regression (LR)

All the predictors are implemented in python using Scikit-learn library and Keras with tensorflow back end for implementing neural network regression model. Each model is trained with a set of features exracted from the crisper-cas9 dataset including:

Efficiency, Specificity, BS_Length, and Distance_exon_BS (D)

