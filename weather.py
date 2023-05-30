#Weather prediction using ML GaussianNB Model
import pandas as pd
wt=pd.read_csv('weather.csv')
wt.head()
wt.describe()
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
numerics=LabelEncoder()
inputs=wt.drop('play',axis='columns')
target=wt['play']
print(target)
inputs['outlook_n']=numerics.fit_transform(inputs['outlook'])
inputs['temperature_n']=numerics.fit_transform(inputs['temperature'])
inputs['humidity_n']=numerics.fit_transform(inputs['humidity'])
inputs['windy_n']=numerics.fit_transform(inputs['windy'])
print(inputs)
inputs_n=inputs.drop(['outlook','temperature','humidity','windy'],axis='columns')
print(inputs_n)
classifier=GaussianNB()
classifier.fit(inputs_n,target)
classifier.score(inputs_n,target)
weather=classifier.predict([[1,1,0,0]])
print(weather)
if weather[0]=='yes':
  print("The game will happens tomorrow")
else:
  print("The game will not happens tomorrow")