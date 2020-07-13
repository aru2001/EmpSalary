import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
data = pd.read_csv('db.csv')

feature = ['exp.','cgpa(10)','cgpa(12)','cgpa(be)']

x = data[feature]
y = data['salary']

model = LinearRegression()
model.fit(x,y)

pickle.dump(model,open('model.pkl','wb'))
pickle.load(open('model.pkl','rb'))