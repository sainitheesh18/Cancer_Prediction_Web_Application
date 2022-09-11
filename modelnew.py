import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn import preprocessing
le=preprocessing.LabelEncoder()
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import log_loss, f1_score
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.naive_bayes import GaussianNB


data=pd.read_csv('Cancer.csv')

data=data.drop(['Patient Id'],axis=1)
data=data.drop(['Gender'],axis=1)
data=data.drop(['Age'],axis=1)

data['Level'].replace('Medium','High',inplace=True)
data['Level'].replace('High','1',inplace=True)
data['Level'].replace('Low','0',inplace=True)
data.Level=pd.to_numeric(data['Level'])





col=[ 'AirPollution', 'Alcoholuse', 'DustAllergy',
       'OccuPationalHazards', 'GeneticRisk', 'chronicLungDisease',
       'BalancedDiet', 'Obesity', 'Smoking', 'PassiveSmoker', 'ChestPain',
       'CoughingofBlood', 'Fatigue', 'WeightLoss', 'ShortnessofBreath',
       'Wheezing', 'SwallowingDifficulty', 'ClubbingofFingerNails',
       'FrequentCold', 'DryCough', 'Snoring']
for c in col:
  data[c]=data[c]/data[c].abs().max()       

# create the data
X = data.drop('Level',axis = 1)
y = data['Level']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30)



from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=0).fit(X_train, y_train)
y_log=model.predict(X_test)
print('Accuracy score : ', accuracy_score(y_test,y_log)*100)


# Saving model to disk
pickle.dump(model, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))

