import numpy as np
from flask import Flask,render_template,request
import pickle
app=Flask(__name__,template_folder='template')

model = pickle.load(open('model.pkl','rb'))

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/pre",methods=['post'])
def prediction():
    feature = [int(x) for x in request.form.values()]
    arr = [np.array(feature)]
    predi = model.predict(arr)
    output = round(predi[0],2)
    return render_template('home.html',predict_text = "emp salary is {}".format(output))

if __name__=="__main__":
    app.run(debug=True)