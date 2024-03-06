import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import mean_squared_error
import datetime

data = pd.read_csv('bussiness.csv')

x = data[['Days', 'Instock']]
y = data['Sellstock']

PDays=3;
PInstock=1381086;
PSellStock=0;
PProfit=0;

#X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#Train Data regressor Model+++++++++++++++++++++++++++
regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
newtrain = np.hsplit(X_train, 2)
X_trainnew=newtrain[1]
plt.scatter(X_trainnew, y_train, color = 'blue')

#Test Data Model+++++++++++++++++++++++++++

ytest_pred = regressor.predict(X_test)
newtest = np.hsplit(X_test, 2)
X_testnew=newtest[1]
plt.scatter(X_testnew,ytest_pred,color = 'green')

#Set New predict value+++++++++++++++++++++++++++

newpredicttest = np.array([PDays,PInstock]).reshape(1, 2)
Ynewtest_pred = regressor.predict(newpredicttest)
print("Business Sell Stock In Next ",PDays ," Days is")
print(Ynewtest_pred[0])
PSellStock=Ynewtest_pred[0]
newptest = np.hsplit(newpredicttest, 2)
X_testnewpredict=newptest[1]
plt.scatter(X_testnewpredict,Ynewtest_pred,color = 'red')

plt.title('Business Sell Forecast')
plt.xlabel('Instock')
plt.ylabel('Sellstock')
plt.show()

###################################################################
x = data[['Days', 'Instock','Sellstock']]
y = data['profit']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#Train Data regressor Model+++++++++++++++++++++++++++
regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
newtrain = np.hsplit(X_train, 3)
X_trainnew=newtrain[2]
plt.scatter(X_trainnew, y_train, color = 'blue')

#Test Data Model+++++++++++++++++++++++++++

ytest_pred = regressor.predict(X_test)
newtest = np.hsplit(X_test, 3)
X_testnew=newtest[2]
plt.scatter(X_testnew,ytest_pred,color = 'green')

#Set New predict value+++++++++++++++++++++++++++

newpredicttest = np.array([PDays,PInstock,PSellStock]).reshape(1, 3)
Ynewtest_pred = regressor.predict(newpredicttest)
print("Profit Of Sell Stock In Next ",PDays ," Days is")
print(Ynewtest_pred[0])
newptest = np.hsplit(newpredicttest, 3)
X_testnewpredict=newptest[2]
plt.scatter(X_testnewpredict,Ynewtest_pred,color = 'red')


plt.title('Business Profit Forecast')
plt.xlabel('Sellstock')
plt.ylabel('Profit')
plt.show()

    
