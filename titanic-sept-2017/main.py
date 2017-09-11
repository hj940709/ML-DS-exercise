cd f:/python_project/ML-DS-exercise/titanic-sept-2017

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier

train_raw = pd.read_csv("./train.csv")
test_raw = pd.read_csv("./test.csv")

def preprocess(df):
    df = df.drop(["Name","Ticket","PassengerId","Cabin"],axis=1)
    df["Sex"] = df["Sex"].astype('category').cat.codes
    df["Embarked"] = df["Embarked"].astype('category').cat.codes
    df["Pclass"] = df["Pclass"].fillna(df["Pclass"].mode())
    df["Sex"] = df["Sex"].fillna(df["Sex"].mode())
    df["Age"] = df["Age"].fillna(int(np.ceil(df["Age"].mean())))
    df["SibSp"] = df["SibSp"].fillna(df["SibSp"].mode())
    df["Parch"] = df["Parch"].fillna(df["Parch"].mode())
    df["Fare"] = df["Fare"].fillna(df["Fare"].mean())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode())
    '''
    deckmdoe = int(df["Deck"].map(lambda x:x==-1 and np.nan or x).mode())
    df["Deck"] = df["Deck"].map(lambda x:x==-1 and deckmdoe or x)
    '''
    return df

train = preprocess(train_raw)
test = preprocess(test_raw)
train["Survived"] = train["Survived"].fillna(train["Survived"].mode())



y=np.array(train["Survived"])
trainX=np.array(train.drop("Survived",axis=1))
testX=np.array(test)
svm_clf = svm.SVC(probability=True)
svm_clf.fit(trainX, y)
lr_clf = LogisticRegression(random_state=1)
lr_clf.fit(trainX,y)
NB_clf = GaussianNB()
NB_clf.fit(trainX,y)
rf_clf = RandomForestClassifier(random_state=1)
rf_clf.fit(trainX,y)


eclf =  VotingClassifier(estimators=[('svc', svm_clf),('lr', lr_clf),
            ('rf',rf_clf),('NB', NB_clf)], voting='soft', weights=[5,6,9,1])

for clf, label in zip([svm_clf,lr_clf, rf_clf, NB_clf, eclf],
                      ['Support Vector','Logistic Regression',
                       'Random Forest', 'naive Bayes', 'Ensemble']):
    scores = cross_val_score(clf, trainX, y, cv=10, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

eclf.fit(trainX,y)
pred = eclf.predict(np.array(trainX))
(pred==y).sum()/y.size

result = pd.DataFrame()
result["PassengerId"] = test_raw["PassengerId"]
result["Survived"] = eclf.predict(np.array(testX))

result.to_csv("./submission.csv",index=False)