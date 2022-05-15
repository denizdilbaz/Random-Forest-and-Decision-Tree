# -*- coding: utf-8 -*-
"""
Created on Sun May 15 22:11:12 2022

@author: 05414015011
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

veriler = pd.read_csv('creditcard.csv')           # verileri yükledik

X = veriler.drop(["Class"],axis = 1)
y = veriler.Class


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 0, stratify = y )

clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train,y_train)
test_sonucu = clf.predict(X_test)
print('Karar ağacı doğruluk değeri: ' + str(accuracy_score(test_sonucu, y_test)))


kararagacimatris = confusion_matrix(y_test,test_sonucu)
print(kararagacimatris)

plt.matshow(kararagacimatris)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('Gerçekler')
plt.xlabel('Tahminler')
plt.show()


Pre = kararagacimatris[0][0] / (kararagacimatris[0][0] + kararagacimatris[0][1])
print("K-fold cross validation ile hesaplanmadan önce confusion matrix'ten elde edilen Precision:",Pre)
Rec = kararagacimatris[0][0] / (kararagacimatris[0][0] + kararagacimatris[1][0])
print("K-fold cross validation ile hesaplanmadan önce confusion matrix'ten elde edilen Recall:",Rec)



scores = cross_val_score(clf,X_train,y_train,cv = 10)
print("Accuracy:",scores.mean())

recall = cross_val_score(clf, X_train,y_train, cv=10, scoring='recall')
print('Recall', np.mean(recall))

precision = cross_val_score(clf, X_train, y_train, cv=10, scoring='precision')
print('Precision', np.mean(precision))

test = ([172792.0,-1.0123123,0.123123,2.198712,1.992763,-3.8765123,-0.303129,-0.881234,-0.501954,1.081976,1.519876,-0.761232,-0.649087,0.151617,-0.181232,0.8519892,-0.921032,-0.3819203,-0.1933423,-0.04,-0.231,-0.09123,0.2198102,-1.719234,0.87102,1.112122,-0.65210,-1.202876,0.343526,128.99],[172792.0,1.284212,-2.58130,1.30281,-0.72123,2.129982,-2.29123,2.403131,-0.50,0.08052,0.551023,-1.10413,-0.648721,0.41818,0.41123,1.12858,-0.29134,0.31328,0.1098,0.213312,1.85911,1.15987,-0.223113,0.37171,1.02189,1.46123,0.82178,0.2113,-0.653333,31.99])
yeni_girdi = clf.predict(test)
print("1.Verinin Sınıfı {0}, 2.Verinin Sınıfı {1} ".format(yeni_girdi[0],yeni_girdi[1]))



rf = RandomForestClassifier(n_estimators=100, max_depth = 3, random_state=42)
rf.fit(X_train, y_train)
prediction = rf.predict(X_test)
print('Random Forest doğruluk değeri: ' + str(accuracy_score(prediction, y_test)))

randomforestmatris = confusion_matrix(y_test,prediction)
print(randomforestmatris)


plt.matshow(randomforestmatris)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('Gerçekler')
plt.xlabel('Tahminler')
plt.show()




PreRandom = randomforestmatris[0][0] / (randomforestmatris[0][0] + randomforestmatris[0][1])
print("K-fold cross validation ile hesaplanmadan önce confusion matrix'ten elde edilen Precision:",PreRandom)
RecRandom = randomforestmatris[0][0] / (randomforestmatris[0][0] + randomforestmatris[1][0])
print("K-fold cross validation ile hesaplanmadan önce confusion matrix'ten elde edilen Recall:",RecRandom)



scoresRandom = cross_val_score(rf,X_train,y_train,cv = 10)
print("Accuracy:",scores.mean())

recallRandom = cross_val_score(rf, X_train,y_train, cv=10, scoring='recall')
print('Recall', np.mean(recallRandom))

precisionRandom = cross_val_score(rf, X_train, y_train, cv=10, scoring='precision')
print('Precision', np.mean(precisionRandom))

yeni_girdiRandomForest = rf.predict(test)
print("1.Verinin Sınıfı {0}, 2.Verinin Sınıfı {1} ".format(yeni_girdiRandomForest[0],yeni_girdiRandomForest[1]))

