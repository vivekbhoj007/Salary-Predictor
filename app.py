# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 12:50:50 2020

@author: vbhoj
"""

#pip install flask
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')
        # return "<h1>Welcome to homepage</h1>"


@app.route('/predict', methods=['POST','GET'])
def predict():
    
    val = request.form['experience']
    val = np.array(int(val))
    final_features = val.reshape(1,1)
    prediction = model.predict(final_features)
    output = np.round(prediction[0],2)

    if output is ():
        return "There is some error"
    else:
        return render_template('index.html', prediction_text = 'Employee salary should be {}'.format(output))

# app.run()

if __name__ == "__main__":
    app.run(debug = True)
       
    
    