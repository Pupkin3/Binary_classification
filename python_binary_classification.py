# -*- coding: utf-8 -*-
"""
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def Model(filename, model_name = 'DecisionTree'):
    
    # return model

    # reading file with dataset
    df = pd.read_csv(filename)

    # one hot encoding
    df_dummies = pd.get_dummies(df)
    features = df_dummies.ix[:, :-2]
    X = features.values
    y = df_dummies['income_>50K']

    # make train and test datasets 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    
    if (model_name == 'DecisionTree'):
        tree = DecisionTreeClassifier(max_depth = 5)
        tree.fit(X_train, y_train)
        print("Decision Tree score:\nTrain => ", tree.score(X_train, y_train),"\nTest => " ,tree.score(X_test, y_test))
        return tree
       
    if (model_name == 'LogReg'): 
        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)
        print("Logistic Regression score:\nTrain => ", logreg.score(X_train, y_train),"\nTest => " ,logreg.score(X_test, y_test))
        return logreg

    if (model_name == 'RandomForest'): 
        forest = RandomForestClassifier(n_estimators = 30)
        forest.fit(X_train, y_train)
        print("Random Forest score:\nTrain => ", forest.score(X_train, y_train),"\nTest => " ,forest.score(X_test, y_test))
        return forest

if __name__ == "__main__": print("Model(filename, model_name)")