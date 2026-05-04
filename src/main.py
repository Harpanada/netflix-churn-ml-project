import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

model=joblib.load('../models/Trained_Dec_tree.pkl')
app = Flask(__name__)


@app.get("/")
def root():
   return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
   raw_input=request.json
   
   df=pd.DataFrame([raw_input])

   result=model.predict(df)
   pred=result[0]
    
   if pred == 0:
      pred= 'Not Churned'
   elif pred == 1:
      pred= 'Churned'

   return jsonify ({
      'prediction':pred
      })


